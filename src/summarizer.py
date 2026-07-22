from typing import List, Tuple, Dict
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from src.config import OPTIMAL_HYPERPARAMS


def compute_k_adaptive(n_sentences: int, summary_length: str = 'medium', enable_buffer: bool = True) -> int:
    """
    Tính toán số cụm K thích ứng có kèm theo hệ số đệm lọc trùng (Buffer K).
    """
    if n_sentences < 4:
        return n_sentences

    alpha = OPTIMAL_HYPERPARAMS['alpha']
    scales = {'brief': 0.7, 'medium': 1.0, 'detailed': 1.4}
    scale = scales.get(summary_length, 1.0)

    target_k = int(round(n_sentences * alpha * scale))
    target_k = max(2, min(10, target_k))

    if enable_buffer and n_sentences > 6:
        kmeans_k = target_k + OPTIMAL_HYPERPARAMS['buffer_k']
    else:
        kmeans_k = target_k

    return min(n_sentences, kmeans_k)


def kmeans_cluster(sentences: List[Tuple[int, str]], embeddings: np.ndarray, k: int) -> Tuple[List[int], List[str], List[np.ndarray], float]:
    """
    Thực hiện phân cụm K-Means và chọn câu nằm gần tâm cụm nhất.
    Tính toán Chỉ số Nội tại (Intrinsic Metric): Silhouette Score.
    """
    k = min(k, len(sentences))
    if k <= 0 or len(sentences) == 0:
        return [], [], [], 0.0

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(embeddings)
    centroids = kmeans.cluster_centers_

    # Tính chỉ số Silhouette Score (Nội tại)
    sil_score = 0.0
    if 1 < k < len(sentences):
        try:
            sil_score = float(silhouette_score(embeddings, kmeans.labels_))
        except Exception:
            sil_score = 0.0

    selected_indices = []
    selected_sentences = []
    selected_embeddings = []

    for cluster_idx in range(k):
        mask = (kmeans.labels_ == cluster_idx)
        if not np.any(mask):
            continue

        cluster_embs = embeddings[mask]
        cluster_sents = [sentences[i] for i, m in enumerate(mask) if m]

        centroid = centroids[cluster_idx].reshape(1, -1)
        sims = cosine_similarity(cluster_embs, centroid).flatten()
        best_idx = int(np.argmax(sims))

        selected_indices.append(cluster_sents[best_idx][0])
        selected_sentences.append(cluster_sents[best_idx][1])
        selected_embeddings.append(cluster_embs[best_idx])

    return selected_indices, selected_sentences, selected_embeddings, sil_score


def filter_redundant(indices: List[int], sents: List[str], embs: List[np.ndarray], threshold: float = None) -> Tuple[List[int], List[str], List[np.ndarray]]:
    """
    Bước lọc sau (Post-filtering) loại bỏ các câu trùng lặp ngữ nghĩa dựa trên ngưỡng Cosine Similarity.
    """
    if threshold is None:
        threshold = OPTIMAL_HYPERPARAMS['theta']

    keep_indices = []
    for i in range(len(sents)):
        is_redundant = False
        for j in keep_indices:
            sim = cosine_similarity([embs[i]], [embs[j]])[0][0]
            if sim > threshold:
                is_redundant = True
                break
        if not is_redundant:
            keep_indices.append(i)

    return (
        [indices[i] for i in keep_indices],
        [sents[i] for i in keep_indices],
        [embs[i] for i in keep_indices]
    )


def reorder_by_original(indices: List[int], sents: List[str]) -> List[str]:
    """
    Sắp xếp lại các câu tóm tắt được chọn về đúng thứ tự vị trí xuất hiện ban đầu trong bài báo.
    """
    paired = sorted(zip(indices, sents), key=lambda x: x[0])
    return [s for _, s in paired]


def diversity_score(embeddings: List[np.ndarray]) -> float:
    """
    Tính chỉ số Đa dạng (Diversity Score = 1 - trung bình cộng Cosine Similarity giữa các cặp câu).
    """
    n = len(embeddings)
    if n < 2:
        return 1.0
    sims = [
        cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
        for i in range(n) for j in range(i + 1, n)
    ]
    return float(round(1.0 - np.mean(sims), 4))
