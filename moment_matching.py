import numpy as np
from itertools import combinations_with_replacement
from typing import List, Tuple, Optional
from math import comb

def compute_Nmk(m: int, k: int) -> int:
    """
    Args:
        m: 每个对象的维度（如 d_in + d_out）。
        k: 需要匹配的最高矩阶数。

    Returns:
        组合数 C(m+k, k)。
    """
    return comb(m + k, k)


def get_monomial_patterns(m: int, k: int) -> List[Tuple[int, ...]]:
    """预计算 φ(u) 中所有单项式的指数模式。

    φ(u) 包含所有满足 Σ_i d_i ≤ k 的单项式 u_1^{d_1}·...·u_m^{d_m}。
    每个模式是一个 m 元组 (d_1, ..., d_m)，总数为 N_{m,k} = C(m+k, k)。

    Args:
        m: 对象维度。
        k: 最高矩阶数。

    Returns:
        指数元组列表，长度 N_{m,k}，以常数项（全零）开头。
    """
    patterns: List[Tuple[int, ...]] = []

    # 0 阶：常数项 1
    patterns.append(tuple([0] * m))

    for r in range(1, k + 1):
        # r 个索引（可重复）的每种组合对应一个 r 阶单项式
        for combo in combinations_with_replacement(range(m), r):
            exp = [0] * m
            for idx in combo:
                exp[idx] += 1
            patterns.append(tuple(exp))

    assert len(patterns) == compute_Nmk(m, k), \
        f"模式数 {len(patterns)} ≠ N_{{m,k}} = {compute_Nmk(m, k)}"
    return patterns


# ==============================================================================
# 2. 特征映射 φ(u)
# ==============================================================================

def build_phi(u: np.ndarray, k: int,
              patterns: Optional[List[Tuple[int, ...]]] = None) -> np.ndarray:
    """对单个对象 u ∈ R^m 构建特征映射向量 φ(u)。

    φ(u) = vect(1, u, u^{⊗2}, ..., u^{⊗k})
    即 u 的所有 ≤ k 阶单项式组成的向量，长度为 N_{m,k}。

    Args:
        u: 对象向量，形状 (m,)。
        k: 单项式最高阶数。
        patterns: 预计算的单项式模式。为 None 时自动根据 m=len(u) 计算。

    Returns:
        φ(u)，形状为 (N_{m,k},) 的 numpy 数组。
    """
    m = len(u)
    if patterns is None:
        patterns = get_monomial_patterns(m, k)

    phi = np.ones(len(patterns), dtype=np.float64)
    for i, exp in enumerate(patterns):
        if i == 0:
            continue  # 常数项，phi[0] = 1
        val = 1.0
        for dim_idx, power in enumerate(exp):
            if power > 0:
                val *= u[dim_idx] ** power
        phi[i] = val

    return phi


def build_feature_matrix(objects: np.ndarray, k: int,
                         patterns: Optional[List[Tuple[int, ...]]] = None) -> np.ndarray:
    """构建特征矩阵 A，其每一列为对应对象的 φ(u_i)。

    A 的形状为 (N_{m,k}, N)，满足 A·c = moments，其中 c 为权重向量。
    匹配矩即意味着找到新权重 c' 使 A·c = A·c'。

    Args:
        objects: 簇矩阵，形状 (N, m)，每行为一个 u_i。
        k: 矩匹配阶数。
        patterns: 预计算的单项式模式。

    Returns:
        特征矩阵 A，形状 (N_{m,k}, N)。
    """
    N, m = objects.shape
    if patterns is None:
        patterns = get_monomial_patterns(m, k)

    Nmk = len(patterns)
    A = np.zeros((Nmk, N), dtype=np.float64)

    for j in range(N):
        A[:, j] = build_phi(objects[j], k, patterns)

    return A


def find_null_vector(A: np.ndarray) -> np.ndarray:
    """求 v 使得 A·v ≈ 0（A 的零空间中的一个非零向量）。

    使用 SVD 分解：最小奇异值对应的右奇异向量给出最佳近似零空间方向。
    对于胖矩阵（列数 > 行数），零空间维数至少为 列数 - 行数。

    保证 v 中至少有一个分量为正（用于计算更新步长 t）。

    Args:
        A: 矩阵，形状 (Nmk, N)，满足 N > Nmk。

    Returns:
        零向量 v，形状 (N,)，满足 A·v ≈ 0 且 max(v) > 0。
    """
    # 完全 SVD 分解：对 (p×q) 胖矩阵 (p<q)，Vh 形状为 (q,q)
    # 最后 (q-p) 个右奇异向量张成零空间
    _, _, Vt = np.linalg.svd(A, full_matrices=True)

    # 取最小奇异值对应的右奇异向量（Vt 最后一行 = V 最后一列）
    v = Vt[-1, :].copy()

    # 保证至少一个分量为正（若全 ≤0 则取反方向，两者都在零空间中）
    if np.all(v <= 0):
        v = -v

    return v


def moment_matching_reduce(
    weights: np.ndarray,
    objects: np.ndarray,
    k: int,
    m: int,
    patterns: Optional[List[Tuple[int, ...]]] = None,
    tol: float = 1e-12
) -> np.ndarray:
    """算法2：在保持前 k 阶矩不变的条件下，将加权对象数削减至 ≤ N_{m,k}。

    给定 N 个加权对象 {(c_i, u_i)}，其中 N > N_{m,k}，本算法迭代地消去
    对象（将其权重置零），同时保持前 k 阶张量矩不变：

        Σ_i c_i · φ(u_i) = Σ_i c'_i · φ(u_i)

    每轮迭代:
      1. 对活跃对象建立特征矩阵 A（列为 φ(u_j)）。
      2. 用 SVD 求 A 零空间中的方向 v。
      3. 计算步长 t = min_{j: v_j > 0} (c_j / v_j)。
      4. 更新 c ← c - t·v，至少一个权重变为 0。

    该过程保证最终非零权重数 ≤ N_{m,k}。

    Args:
        weights: 当前权重，形状 (N,)，第 i 个元素 c_i ≥ 0。
        objects: 对象向量矩阵，形状 (N, m)，第 i 行为 u_i ∈ R^m。
        k: 矩匹配阶数。
        m: 每个对象的维度。
        patterns: 预计算的 φ 单项式模式。
        tol: 判定权重为零的容差。

    Returns:
        新的权重数组，形状 (N,)，非零项 ≤ N_{m,k}，且保持前 k 阶矩不变。
    """
    N = len(weights)
    Nmk = compute_Nmk(m, k)

    # 已足够小，无需压缩
    if N <= Nmk:
        return weights.copy()

    if patterns is None:
        patterns = get_monomial_patterns(m, k)

    c = weights.copy().astype(np.float64)
    c = np.maximum(c, 0.0)  # 确保非负

    iteration = 0
    max_iterations = N  # 安全上限：每次至少消去一个对象

    while np.sum(c > tol) > Nmk and iteration < max_iterations:
        iteration += 1
        active_idx = np.where(c > tol)[0]

        # —— 构建活跃对象的特征矩阵 A ——
        A = build_feature_matrix(objects[active_idx], k, patterns)

        # 列归一化以增强数值稳定性
        col_norms = np.linalg.norm(A, axis=0)
        col_norms[col_norms < 1e-15] = 1.0
        A_normed = A / col_norms[np.newaxis, :]

        # —— 寻找零空间方向 ——
        v = find_null_vector(A_normed)

        # 将 v 映射回未归一化空间
        v = v / col_norms

        if np.all(v <= tol):
            v = -v

        # —— 计算步长 t ——
        pos_mask = v > tol
        if not np.any(pos_mask):
            break

        ratios = np.full_like(v, np.inf)
        ratios[pos_mask] = c[active_idx][pos_mask] / v[pos_mask]
        t = np.min(ratios)

        if t <= tol or np.isinf(t):
            break

        # —— 更新权重：至少一个变为 0 ——
        c[active_idx] = c[active_idx] - t * v

        # 清理接近零的权重
        c[np.abs(c) < tol] = 0.0
        c = np.maximum(c, 0.0)

    # 最终清理
    c[np.abs(c) < tol] = 0.0
    c = np.maximum(c, 0.0)

    return c
