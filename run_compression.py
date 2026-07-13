"""
CNN 卷积核压缩实验主脚本
=========================
基于矩匹配（Moment Matching）对预训练 CNN 模型逐层压缩，
评估不同压缩率、不同 k 值对模型输出的影响。

输出：
- 每层的 MSE vs 压缩强度曲线（不同 k 值）
- 每层的最佳压缩效果统计
"""

import torch
import copy
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# 复用已有模块
import DataLoader
import parameters
from model_32 import Model_32

# 压缩模块
from compression import compress_sequential_layer


# ==============================================================================
# 0. 辅助：设置中文字体（matplotlib）
# ==============================================================================

def setup_chinese_font():
    """尝试设置 matplotlib 中文字体，使图表可以显示中文标签。"""
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei',
                                            'DejaVu Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        pass


setup_chinese_font()


# ==============================================================================
# 1. 模型评估
# ==============================================================================

@torch.no_grad()
def evaluate_model(model, eval_batches, device):
    """
    评估模型在给定数据上的输出和准确率。

    Args:
        model: PyTorch 模型（eval 模式）
        eval_batches: 列表，每个元素是 (images, labels)
        device: 计算设备

    Returns:
        all_logits: 形状 (N_total, 10) 的张量
        accuracy: 浮点数，分类准确率
    """
    model.eval()
    all_logits = []
    total_correct = 0
    total_samples = 0

    for images, labels in eval_batches:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)  # (batch, 10)

        all_logits.append(logits.cpu())

        # 准确率
        _, preds = logits.max(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    all_logits = torch.cat(all_logits, dim=0)  # (N_total, 10)
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    return all_logits, accuracy


def compute_mse_between_logits(logits1: torch.Tensor,
                                logits2: torch.Tensor) -> float:
    """
    计算两组 logits 之间的均方误差（MSE）。

    Args:
        logits1: 原始模型输出 (N, classes)
        logits2: 压缩模型输出 (N, classes)

    Returns:
        mse: 平均 MSE（标量）
    """
    return torch.mean((logits1 - logits2) ** 2).item()


# ==============================================================================
# 2. 数据准备
# ==============================================================================

def prepare_eval_batches(num_batches: int = 100):
    """
    准备评估用的数据 batch。

    为了节省时间，只取训练集的前 num_batches 个 batch。
    固定 batch 顺序（shuffle=False 或使用固定的 DataLoader），
    确保原始模型和压缩模型评估的是同一批数据。

    Args:
        num_batches: 取的 batch 数量

    Returns:
        eval_batches: 列表，每个元素是 (images, labels)
    """
    params = parameters.get_parameters()
    trainset = DataLoader.Full_train_data_Loader(params)

    eval_batches = []
    for i, (images, labels) in enumerate(trainset):
        if i >= num_batches:
            break
        eval_batches.append((images, labels))

    print(f"准备评估数据：{len(eval_batches)} 个 batch，"
          f"batch_size={params.batch_size}，"
          f"总计约 {len(eval_batches) * params.batch_size} 张图片")
    return eval_batches


# ==============================================================================
# 3. 模型加载
# ==============================================================================

def load_pretrained_model(device, weights_path="model_weights.pth"):
    """
    加载预训练模型。

    优先从指定路径加载权重；若文件不存在，则初始化一个随机权重的模型
    （仅用于测试代码流程）。

    自动处理以下情况：
    - DataParallel 保存的权重（带 "module." 前缀）
    - 不同 PyTorch 版本的兼容性
    - BN 层 num_batches_tracked 等额外 key

    Args:
        device: 计算设备
        weights_path: 权重文件路径

    Returns:
        model: 加载了权重的模型（eval 模式）
    """
    model = Model_32().to(device)

    if os.path.exists(weights_path):
        # 兼容不同 PyTorch 版本
        try:
            state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        except TypeError:
            state_dict = torch.load(weights_path, map_location=device)

        # 处理 DataParallel 的 "module." 前缀
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            print(f"已从 {weights_path} 加载预训练权重（去除 module. 前缀）")
        else:
            print(f"已从 {weights_path} 加载预训练权重")

        # 忽略 num_batches_tracked 等不在模型中的 key（strict=False 时自动忽略）
        # 但也要检测真正的 missing/unexpected keys
        model_state_keys = set(model.state_dict().keys())
        loaded_keys = set(state_dict.keys())
        extra_keys = loaded_keys - model_state_keys
        missing_keys = model_state_keys - loaded_keys

        # 过滤掉无害的 extra key
        harmless_extra = {k for k in extra_keys if "num_batches_tracked" in k}
        extra_keys = extra_keys - harmless_extra

        if extra_keys:
            print(f"  注意：权重文件中多余的 key: {extra_keys}")
        if missing_keys:
            print(f"  警告：模型中缺少的 key: {missing_keys}")

        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"警告：未找到 {weights_path}，使用随机初始化权重")
        print("请先运行 train_full.py 训练模型，或使用 train_full() 函数训练")

    model.eval()
    return model


# ==============================================================================
# 4. 主实验循环
# ==============================================================================

def run_compression_experiment():
    """
    主实验：遍历每一层、每个 k 值、每个压缩率，评估压缩效果。
    """
    # --- 设备 ---
    params = parameters.get_parameters()
    device = params.device if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # --- 加载模型和数据 ---
    pretrained_model = load_pretrained_model(device)
    eval_batches = prepare_eval_batches(num_batches=100)

    # --- 预计算原始模型输出（基准）---
    print("\n计算原始模型输出（基准）...")
    original_logits, original_acc = evaluate_model(
        pretrained_model, eval_batches, device,
    )
    print(f"原始模型准确率: {original_acc:.4f}")

    # --- 超参数 ---
    ratios = [0.9, 0.85, 0.8, 0.7, 0.5]
    k_values = [2, 3, 4, 5, 6]
    layers_to_compress = ['Layer1', 'Layer2', 'Layer3', 'Layer4']

    # 存储结构: results[layer_name][k] = [(ratio, mse, acc_drop), ...]
    results = {
        layer: {k: [] for k in k_values}
        for layer in layers_to_compress
    }
    # 最佳 k 值: best_k[layer_name] = (best_k, best_avg_mse)
    best_k = {}

    # --- 遍历每一层 ---
    for layer_name in layers_to_compress:
        print(f"\n{'=' * 60}")
        print(f"处理 {layer_name}")
        print(f"{'=' * 60}")

        for k in k_values:
            print(f"  k={k}: ", end="", flush=True)

            for ratio in ratios:
                # 深拷贝原始模型，只压缩当前层
                compressed_model = copy.deepcopy(pretrained_model)
                target_layer = getattr(compressed_model, layer_name)

                # 执行压缩
                compress_sequential_layer(target_layer, k, ratio)

                # 评估
                compressed_logits, compressed_acc = evaluate_model(
                    compressed_model, eval_batches, device,
                )
                mse = compute_mse_between_logits(original_logits, compressed_logits)
                acc_drop = original_acc - compressed_acc

                results[layer_name][k].append((ratio, mse, acc_drop))
                print(f"[r={ratio:.2f}] {mse:.6f}", end=" ", flush=True)

            print()  # 换行

        # 找出该层中每个 ratio 下表现最好的 k 值（按 MSE 平均）
        # 先计算每个 k 的平均 MSE
        k_avg_mse = {}
        for k in k_values:
            mse_list = [r[1] for r in results[layer_name][k]]
            k_avg_mse[k] = np.mean(mse_list)

        best_k_for_layer = min(k_avg_mse, key=k_avg_mse.get)
        best_k[layer_name] = (best_k_for_layer, k_avg_mse[best_k_for_layer])
        print(f"  >> {layer_name} 最佳 k={best_k_for_layer}, "
              f"平均 MSE={k_avg_mse[best_k_for_layer]:.6f}")

    # --- 输出汇总 ---
    print(f"\n{'=' * 60}")
    print("汇总：各层最佳矩匹配阶数")
    print(f"{'=' * 60}")
    for layer_name in layers_to_compress:
        k, mse = best_k[layer_name]
        print(f"  {layer_name}: 最佳 k={k}, 平均 MSE={mse:.6f}")

    return results, best_k, original_acc


# ==============================================================================
# 5. 绘图
# ==============================================================================

def plot_results(results, best_k, original_acc, save_dir="./figures"):
    """
    绘制每层的 MSE vs 压缩强度曲线。

    Args:
        results: 实验结果字典
        best_k: 最佳 k 值字典
        original_acc: 原始模型准确率
        save_dir: 图片保存目录
    """
    os.makedirs(save_dir, exist_ok=True)

    k_values = [2, 3, 4, 5, 6]
    layers_to_compress = ['Layer1', 'Layer2', 'Layer3', 'Layer4']

    # 颜色和标记：为每个 k 值分配
    colors = {2: '#1f77b4', 3: '#ff7f0e', 4: '#2ca02c', 5: '#d62728', 6: '#9467bd'}
    markers = {2: 'o', 3: 's', 4: 'D', 5: '^', 6: 'v'}

    # --- 每个 Layer 单独一张图 ---
    for layer_name in layers_to_compress:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        for k in k_values:
            data = results[layer_name][k]
            # 按 ratio 排序
            data_sorted = sorted(data, key=lambda x: x[0])

            ratios_list = [d[0] for d in data_sorted]
            mse_list = [d[1] for d in data_sorted]
            acc_drop_list = [d[2] for d in data_sorted]

            compression_strength = [1 - r for r in ratios_list]  # 压缩强度

            c = colors[k]
            m = markers[k]

            ax1.plot(compression_strength, mse_list,
                     color=c, marker=m, markersize=6, linewidth=1.5,
                     label=f'k={k}')

            ax2.plot(compression_strength, acc_drop_list,
                     color=c, marker=m, markersize=6, linewidth=1.5,
                     label=f'k={k}')

        # 标注最佳 k 值
        best_k_val = best_k[layer_name][0]
        ax1.set_title(f'{layer_name} - Logits MSE\n(Best k={best_k_val})', fontsize=12)
        ax2.set_title(f'{layer_name} - Accuracy Drop\n(Best k={best_k_val})', fontsize=12)

        for ax in (ax1, ax2):
            ax.set_xlabel('Compression Strength (1 - ratio)', fontsize=11)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        ax1.set_ylabel('MSE (Logits)', fontsize=11)
        ax2.set_ylabel('Accuracy Drop', fontsize=11)

        fig.tight_layout()
        save_path = os.path.join(save_dir, f'{layer_name}_compression.png')
        fig.savefig(save_path, dpi=150)
        print(f"图片已保存: {save_path}")
        plt.close(fig)

    # --- 汇总图：所有层的最佳 k 曲线 ---
    fig, ax = plt.subplots(figsize=(10, 6))

    for layer_name in layers_to_compress:
        best_k_val = best_k[layer_name][0]
        data = results[layer_name][best_k_val]
        data_sorted = sorted(data, key=lambda x: x[0])

        ratios_list = [d[0] for d in data_sorted]
        mse_list = [d[1] for d in data_sorted]
        compression_strength = [1 - r for r in ratios_list]

        ax.plot(compression_strength, mse_list,
                marker='o', markersize=7, linewidth=2,
                label=f'{layer_name} (k={best_k_val})')

    ax.set_xlabel('Compression Strength (1 - ratio)', fontsize=12)
    ax.set_ylabel('MSE (Logits)', fontsize=12)
    ax.set_title('Best-k MSE Comparison Across Layers', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_path = os.path.join(save_dir, 'summary_all_layers.png')
    fig.savefig(save_path, dpi=150)
    print(f"图片已保存: {save_path}")
    plt.close(fig)

    # --- 汇总图：所有层的准确率下降 ---
    fig, ax = plt.subplots(figsize=(10, 6))

    for layer_name in layers_to_compress:
        best_k_val = best_k[layer_name][0]
        data = results[layer_name][best_k_val]
        data_sorted = sorted(data, key=lambda x: x[0])

        ratios_list = [d[0] for d in data_sorted]
        acc_drop_list = [d[2] for d in data_sorted]
        compression_strength = [1 - r for r in ratios_list]

        ax.plot(compression_strength, acc_drop_list,
                marker='s', markersize=7, linewidth=2,
                label=f'{layer_name} (k={best_k_val})')

    ax.set_xlabel('Compression Strength (1 - ratio)', fontsize=12)
    ax.set_ylabel('Accuracy Drop', fontsize=12)
    ax.set_title('Best-k Accuracy Drop Across Layers', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_path = os.path.join(save_dir, 'summary_accuracy_drop.png')
    fig.savefig(save_path, dpi=150)
    print(f"图片已保存: {save_path}")
    plt.close(fig)


# ==============================================================================
# 6. 程序入口
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("CNN 卷积核压缩实验 — 基于矩匹配（Moment Matching）")
    print("=" * 70)

    results, best_k, original_acc = run_compression_experiment()

    print(f"\n{'=' * 70}")
    print("生成图表...")
    plot_results(results, best_k, original_acc)

    print("实验完成！")
