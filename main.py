"""
CNN 卷积核压缩实验 — 入口脚本
==============================
基于矩匹配（Moment Matching）对预训练 CNN 模型逐层压缩。
直接运行此文件即可执行完整实验。

使用方式：
    python main.py
    python main.py --device cuda --batch_size 512
"""

import sys
import os

# 确保项目根目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_compression import run_compression_experiment, plot_results

if __name__ == "__main__":
    print("=" * 70)
    print("CNN 卷积核压缩实验 — 基于矩匹配（Moment Matching）")
    print("参考论文：Lottery Ticket Hypothesis & Neural Scaling Laws")
    print("=" * 70)

    results, best_k, original_acc = run_compression_experiment()

    print(f"\n{'=' * 70}")
    print("生成图表...")
    plot_results(results, best_k, original_acc)

    print("\n实验完成！图表已保存至 ./figures/ 目录")
