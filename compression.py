"""
CNN 卷积核压缩模块
===================
基于矩匹配（Moment Matching）的神经网络压缩。

核心思路：
1. 从每个 Basic_block 中提取中间通道对应的"对象向量" u_i
2. 使用 PCA 将高维对象向量降维到低维空间
3. 在低维空间中应用矩匹配算法，选出保留哪些通道
4. 重构 block 权重，将 scales 吸收进 conv2.weight

注意事项：
- downsample（捷径分支）完全不参与压缩
- 每个 block 的输出通道数保持不变
- 压缩的是 conv1 产生的中间状态通道数

参考论文：Lottery Ticket Hypothesis & Neural Scaling Laws
"""

import torch
import numpy as np
from typing import List, Tuple, Optional
from math import comb

# 复用已有的矩匹配模块
from moment_matching import (
    compute_Nmk,
    get_monomial_patterns,
    moment_matching_reduce,
)
from clustering import cluster_active_objects


# ==============================================================================
# 1. 对象提取
# ==============================================================================

def extract_objects_from_block(block) -> List[torch.Tensor]:
    """
    从 Basic_block 中提取所有中间通道对应的对象向量 u_i。

    对于第 i 个中间通道：
        u_i = concat(conv1.weight[i].flatten(), conv2.weight[:, i, :, :].flatten())

    Args:
        block: Basic_block 实例

    Returns:
        objects: 行数为 out_chan 的 Tensor 列表，每个 Tensor 是u_i(一维向量)
    """
    out_chan = block.out_chan
    objects = []

    conv1_weight = block.conv1.weight.data  # shape: [out_chan, in_chan, 3, 3]
    conv2_weight = block.conv2.weight.data  # shape: [out_chan, out_chan, 3, 3]

    for i in range(out_chan):
        # part1: 生成第 i 个通道的卷积核，形状 [in_chan, 3, 3] -> 展平
        part1 = conv1_weight[i].flatten()  # 长度 = in_chan * 3 * 3
        # part2: 消费第 i 个通道的卷积核（所有输出通道的第 i 个输入通道切片）
        part2 = conv2_weight[:, i, :, :].flatten()  # 长度 = out_chan * 3 * 3
        # 拼接
        u_i = torch.cat([part1, part2])
        objects.append(u_i)

    return objects


# ==============================================================================
# 2. PCA 降维（通过 Gram 矩阵，解决 m >> N 的高维问题）
# ==============================================================================

def pca_reduce(
    objects: np.ndarray,
    target_dim: int,
):
    """
    使用 Gram 矩阵技巧对高维对象做 PCA 降维。

    当 m（原始维度）远大于 N（对象数）时，直接计算 m×m 协方差矩阵不可行。
    改为计算 N×N 的 Gram 矩阵 G = U @ U^T，对其特征分解得到主成分。

    Args:
        objects: 形状 (N, m) 的 numpy 数组
        target_dim: 目标降维维度 d, 目标压缩维度最大为N-1

    Returns:
        reduced: 形状 (N, d) 的降维表示
    """
    N, m = objects.shape
    target_dim = max(1, min(target_dim, N - 1, m))

    # 中心化
    mean = objects.mean(axis=0, keepdims=True) #按列取平均
    centered = objects - mean # mean会自动复制成跟objects一样多的行

    # Gram 矩阵 (N, N)
    G = centered @ centered.T

    # 特征分解
    eigenvalues, eigenvectors = np.linalg.eigh(G)

    # 按特征值降序排列
    idx = np.argsort(eigenvalues)[::-1] #从大到小排序
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    eigenvalues = np.maximum(eigenvalues, 0.0)
    total_variance = float(np.sum(eigenvalues))
    if total_variance > 0.0:
        explained_ratios = eigenvalues / total_variance
        cumulative_ratios = np.cumsum(explained_ratios)
    else:
        explained_ratios = np.zeros_like(eigenvalues)
        cumulative_ratios = np.zeros_like(eigenvalues)

    effective_dim = min(target_dim, len(eigenvalues))
    contribution_at_dim = float(cumulative_ratios[effective_dim - 1]) if effective_dim > 0 else 0.0

    threshold_dims = {}
    for threshold in (0.9, 0.75, 0.5):
        if len(cumulative_ratios) == 0:
            threshold_dims[threshold] = 0
            continue
        idx = int(np.searchsorted(cumulative_ratios, threshold, side='left'))
        threshold_dims[threshold] = min(idx + 1, len(cumulative_ratios))

    # 取前 target_dim 个主成分
    eigenvalues = eigenvalues[:target_dim]
    eigenvectors = eigenvectors[:, :target_dim]

    # 降维坐标：Z = S_d @ sqrt(Λ_d)
    # 即每个对象在 d 个主成分方向上的投影坐标
    reduced = eigenvectors * np.sqrt(eigenvalues)  # 得到的 reduced 形状是 (N, d)

    stats = {
        'total_variance': total_variance,
        'explained_ratios': explained_ratios,
        'cumulative_ratios': cumulative_ratios,
        'contribution_at_dim': contribution_at_dim,
        'threshold_dims': threshold_dims,
        'effective_dim': effective_dim,
    }

    print(
        f"PCA 降维: d={target_dim}, 累计贡献度={contribution_at_dim * 100:.2f}%"
    )
    print(
        "PCA 贡献度阈值对应的最小 d: "
        f"90% -> {threshold_dims[0.9]}, "
        f"75% -> {threshold_dims[0.75]}, "
        f"50% -> {threshold_dims[0.5]}"
    )

    reduced = reduced.astype(np.float64)
    return reduced


# ==============================================================================
# 3. 自适应维度选择
# ==============================================================================

def find_optimal_reduced_dim(N: int, target_num: int, k: int,
                              dim_min: int = 2, dim_max: int = 500) -> int:
    """
    选择 PCA 降维维度 d（整个压缩过程中保持不变）。

    唯一约束：N_{d,k} < N（才存在零空间，矩匹配才能压缩）。
    选 d 的标准：abs(N_{d,k} - target_num) 最小，即 N_{d,k} 最接近 target_num。

    N_{d,k} 可以大于或小于 target_num——若大于，矩匹配后取权重前 target_num 个；
    若小于，矩匹配后由 L2 范数补足。

    Args:
        N: 原始通道数
        target_num: 目标保留通道数
        k: 矩匹配阶数
        dim_min: 最小维度
        dim_max: 最大维度（自适应：取 min(N-1, 500)）

    Returns:
        d: 推荐降维维度
    """
    dim_max = min(dim_max, N - 1)
    best_d = dim_min
    best_score = float('inf')

    for d in range(dim_min, dim_max + 1):
        Ndk = compute_Nmk(d, k)
        if Ndk >= N:
            # 无零空间，无法压缩
            continue
        # 越接近 target_num 越好（不论大于还是小于）
        score = abs(Ndk - target_num)
        if score < best_score:
            best_score = score
            best_d = d

    return best_d


# ==============================================================================
# 4. 矩匹配压缩包装函数
# ==============================================================================

def moment_matching_compress(
    objects: np.ndarray,
    target_num: int,
    k: int,
    patterns: Optional[List[Tuple[int, ...]]] = None,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    对 N 个对象向量执行矩匹配压缩，保留 target_num 个。

    维度 d 在整个压缩过程中固定不变（PCA 只做一次）。
    每轮聚类 + 矩匹配后，活跃通道数逐步减少；当 n_active ≤ N_{d,k}
    时（无零空间），若仍超过 target_num 则用 L2 范数挑选。

    Args:
        objects: 形状 (N, m) 的 numpy 数组
        target_num: 目标保留通道数
        k: 矩匹配阶数
        patterns: 预计算的单项式模式
        random_state: 随机种子

    Returns:
        keep_indices: 保留的通道索引（原objects中的行索引）
        scales: 对应的缩放权重
    """
    N = objects.shape[0]
    if target_num >= N:
        return np.arange(N), np.ones(N)

    obj_norms = np.linalg.norm(objects, axis=1)

    # ---- 一次性选定 d，整个过程中不变 ----
    d = find_optimal_reduced_dim(N, target_num, k)
    objects_reduced = pca_reduce(objects, d)  # (N, d)
    Ndk = compute_Nmk(d, k)
    patterns_d = get_monomial_patterns(d, k)

    # ---- 初始化等权重 ----
    current_weights = np.ones(N, dtype=np.float64) / N
    active_idx = np.arange(N) # 生成 [0, 1, 2, ..., N-1]
    tol = 1e-10

    max_iterations = 50
    iteration = 0

    # ---- 迭代聚类 + 矩匹配 ----
    while len(active_idx) > target_num and iteration < max_iterations:
        iteration += 1
        n_active = len(active_idx)

        if n_active <= Ndk:
            # 无零空间，矩匹配无法继续压缩，退出循环
            break

        # 聚类 + 逐簇矩匹配
        clusters = cluster_active_objects(
            objects_reduced, active_idx, Ndk,
            random_state=random_state + iteration,
        )
        new_weights = np.zeros(N, dtype=np.float64)
        for cluster_indices in clusters:
            cluster_weights = moment_matching_reduce(
                current_weights[cluster_indices],
                objects_reduced[cluster_indices],
                k, d,
                patterns=patterns_d,
                tol=tol,
            )
            new_weights[cluster_indices] = cluster_weights
        current_weights = new_weights

        # 更新活跃集
        new_active_idx = np.where(current_weights > tol)[0]
        if len(new_active_idx) >= n_active:
            break  # 没有进展
        active_idx = new_active_idx # 当前还非零的“全局行号集合”

    # ---- 确定最终保留的通道 ----
    mm_kept = np.where(current_weights > tol)[0]
    mm_scales = current_weights[mm_kept]

    if len(mm_kept) >= target_num:
        # 存活数足够，取权重最大的前 target_num
        top = np.argsort(mm_scales)[::-1][:target_num]
        keep_indices = mm_kept[top]
        scales = mm_scales[top]
    else:
        # 存活数不足，保持原样（不补足）
        keep_indices = mm_kept
        scales = mm_scales

    if len(keep_indices) == 0:
        keep_indices = np.array([np.argmax(obj_norms)])
        scales = np.array([1.0])

    return keep_indices, scales


# ==============================================================================
# 5. 权重重构
# ==============================================================================

def reconstruct_block_weights(block, keep_indices: np.ndarray,
                               scales: np.ndarray) -> None:
    """
    根据 keep_indices 和 scales 重构 Basic_block 的权重。

    规则：
    - conv1（输出通道维）：新权重 = 旧权重[keep_indices]，不乘 scales
    - conv2（输入通道维）：新权重 = 旧权重[:, keep_indices] * scales
    - bn1（作用于 conv1 输出）：按 keep_indices 裁剪
    - bn2（作用于 conv2 输出）：输出通道数不变，保持不变
    - ds / bn3（捷径分支）：完全不修改

    Args:
        block: Basic_block 实例（原地修改）
        keep_indices: 保留的通道索引（numpy 数组）
        scales: 缩放权重（numpy 数组）
    """
    device = block.conv1.weight.device
    keep_idx_tensor = torch.from_numpy(keep_indices).long().to(device)
    scales_tensor = torch.from_numpy(scales).float().to(device)

    # --- conv1: 裁剪输出通道 ---
    block.conv1.weight.data = block.conv1.weight.data[keep_idx_tensor, :, :, :]
    if block.conv1.bias is not None:
        block.conv1.bias.data = block.conv1.bias.data[keep_idx_tensor]

    # --- bn1: 裁剪（作用于 conv1 的输出通道）---
    block.bn1.weight.data = block.bn1.weight.data[keep_idx_tensor]
    block.bn1.bias.data = block.bn1.bias.data[keep_idx_tensor]
    block.bn1.running_mean.data = block.bn1.running_mean.data[keep_idx_tensor]
    block.bn1.running_var.data = block.bn1.running_var.data[keep_idx_tensor]
    block.bn1.num_features = len(keep_indices)

    # --- conv2: 裁剪输入通道，乘以 scales ---
    # conv2.weight 形状: [out_chan, in_chan, 3, 3]
    # 裁剪 in_chan 维度（第 1 维）
    block.conv2.weight.data = block.conv2.weight.data[:, keep_idx_tensor, :, :]
    # 乘以 scales：scales 维度 [keep_num]，广播到 [out_chan, keep_num, 3, 3]
    block.conv2.weight.data = block.conv2.weight.data * scales_tensor.view(1, -1, 1, 1)

    # conv2.bias 保持不变（输出通道数不变）

    # --- bn2: 保持不变（conv2 输出通道数未变）---

    # --- ds 和 bn3: 完全不修改 ---
    # （downsample 路径不被压缩影响）


# ==============================================================================
# 6. 单 Block 压缩
# ==============================================================================

def compress_single_block(block, k: int, ratio: float) -> Tuple[int, int, np.ndarray, np.ndarray]:
    """
    压缩单个 Basic_block 的中间通道。

    Args:
        block: Basic_block 实例
        k: 矩匹配阶数
        ratio: 保留比例（如 0.85 表示保留 85% 的中间通道）

    Returns:
        original_num: 压缩前通道数
        kept_num: 压缩后通道数
        keep_indices: 保留的通道索引
        scales: 缩放权重
    """
    # 1. 提取对象向量
    objects_list = extract_objects_from_block(block)
    original_num = len(objects_list) #即最初中间通道的个数

    # 2. 转为 numpy
    objects_np = torch.stack(objects_list).cpu().numpy()  # (N, m)

    # 3. 计算目标通道数
    target_num = max(int(original_num * ratio), 1)

    # 4. 执行矩匹配压缩
    keep_indices, scales = moment_matching_compress(
        objects_np, target_num, k,
    )

    kept_num = len(keep_indices)

    # 5. 重构权重
    reconstruct_block_weights(block, keep_indices, scales)

    return original_num, kept_num, keep_indices, scales


# ==============================================================================
# 7. Layer 压缩
# ==============================================================================

def compress_sequential_layer(layer_module, k: int, ratio: float):
    """
    压缩一个由两个 Basic_block 组成的 Sequential Layer。

    每个 block 独立压缩——block1 和 block2 各自有自己的中间通道。
    downsample 路径完全不参与压缩。

    Args:
        layer_module: nn.Sequential，包含 block1, block2
        k: 矩匹配阶数
        ratio: 保留中间通道的比例
    """
    results = []
    for block in layer_module:
        orig_num, kept_num, keep_indices, scales = compress_single_block(
            block, k, ratio,
        )
        results.append({
            'orig_num': orig_num,
            'kept_num': kept_num,
            'keep_indices': keep_indices,
            'scales': scales,
        })
    return results
