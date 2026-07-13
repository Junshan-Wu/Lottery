"""
Clustering Strategy — 聚类策略
===============================
实现算法1 Step 1 的聚类子程序。

目标：将活跃对象分成多个直径较小的簇（diameter → 0 ⇒ error → 0 per Theorem 3）。
策略：k-means + 单位球面投影，确保每个簇大小略大于 N_{m,k} 以便矩匹配压缩。

论文: "A Universal Compression Theory for Lottery Ticket Hypothesis
       and Neural Scaling Laws"
"""

import numpy as np
from typing import List


# ==============================================================================
# 1. 轻量级 k-means（无外部依赖）
# ==============================================================================

def kmeans_clustering(
    objects: np.ndarray,
    n_clusters: int,
    max_iter: int = 100,
    random_state: int = 42
) -> np.ndarray:
    """轻量级 k-means 聚类实现（不依赖 scikit-learn）。

    对球面上或空间中的对象进行标准 Lloyd 迭代聚类。

    Args:
        objects: 数据矩阵，形状 (N, m)。
        n_clusters: 簇数。
        max_iter: 最大迭代次数。
        random_state: 随机种子。

    Returns:
        簇标签数组，形状 (N,)，取值 [0, n_clusters)。
    """
    N = len(objects)
    if n_clusters >= N:
        return np.arange(N)

    rng = np.random.default_rng(random_state)
    # 随机选取初始质心
    centroid_indices = rng.choice(N, n_clusters, replace=False)
    centroids = objects[centroid_indices].copy().astype(np.float64)
    labels = np.zeros(N, dtype=np.int32)

    for _ in range(max_iter):
        # 分配步：每个点归属到最近质心
        dists = np.zeros((N, n_clusters), dtype=np.float64)
        for cl in range(n_clusters):
            diff = objects - centroids[cl]
            dists[:, cl] = np.sum(diff * diff, axis=1)
        new_labels = np.argmin(dists, axis=1)

        # 更新质心
        new_centroids = np.zeros_like(centroids)
        for cl in range(n_clusters):
            mask = new_labels == cl
            if np.any(mask):
                new_centroids[cl] = objects[mask].mean(axis=0)
            else:
                # 空簇：重新随机初始化
                new_centroids[cl] = objects[rng.choice(N)]

        if np.all(new_labels == labels):
            break

        labels = new_labels
        centroids = new_centroids

    return labels


# ==============================================================================
# 2. 活跃对象聚类（算法1 Step 1 的核心）
# ==============================================================================

def cluster_active_objects(
    objects: np.ndarray,
    active_idx: np.ndarray,
    Nmk: int,
    random_state: int = 42
) -> List[np.ndarray]:
    """对活跃对象聚类，为矩匹配算法准备输入。

    关键设计决策（基于 Theorem 3 的误差界 O(d·r^{k+1})）：
    - 更多的簇 → 每簇直径更小 → 误差更低。
    - 但每簇必须有 > N_{m,k} 个对象才能应用矩匹配。
    - 先将对象投影到单位球面（L2 归一化），使直径上界 ≤ 2，
      满足理论要求的 ||w_i|| ≤ R 有界条件。

    Args:
        objects: 完整对象矩阵，形状 (N_total, m)。
        active_idx: 当前活跃对象（c_i > 0）的索引数组。
        Nmk: N_{m,k}，矩匹配后的最大支撑集大小。
        random_state: 随机种子。

    Returns:
        簇列表，每个簇是 active_idx 中的索引数组。
    """
    n_active = len(active_idx)

    if n_active <= Nmk:
        return [active_idx]

    # —— 投影到单位球面以限制直径 ——
    active_objs = objects[active_idx].copy()
    obj_norms = np.linalg.norm(active_objs, axis=1, keepdims=True)
    obj_norms[obj_norms < 1e-12] = 1.0
    active_objs_normed = active_objs / obj_norms

    # 计算簇数：每簇大小 = Nmk + margin
    margin = max(5, Nmk // 10)
    target_size = Nmk + margin
    n_clusters = max(1, n_active // target_size)

    if n_clusters == 1:
        return [active_idx]

    # 在归一化后的球面上做 k-means
    labels = kmeans_clustering(active_objs_normed, n_clusters,
                               random_state=random_state)

    clusters = []
    for cl in range(n_clusters):
        mask = labels == cl
        cluster_indices = active_idx[mask]
        if len(cluster_indices) > 0:
            clusters.append(cluster_indices)

    return clusters
