import time
import sys
import os
from typing import Dict, Any

# Ensure src module is importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.preprocess import resolve_language, preprocess_text
from src.embedding import embed_sentences
from src.summarizer import compute_k_adaptive, kmeans_cluster, filter_redundant, reorder_by_original, diversity_score


def process_summarization(text: str, user_lang: str = 'auto', length: str = 'medium', use_finetuned: bool = True) -> Dict[str, Any]:
    """
    Service wrapper that integrates Preprocessing, SBERT Embedding, K-Means Clustering,
    Post-filtering, Reordering, and Metrics Calculation.
    """
    start_time = time.time()

    # 1. Resolve Language (Auto Detect)
    lang = resolve_language(text, user_choice=user_lang)

    # 2. Sentence Tokenization & Preprocessing
    sentences = preprocess_text(text, lang=lang)
    if not sentences:
        raise ValueError("Văn bản quá ngắn hoặc không chứa câu hợp lệ để tóm tắt.")

    # 3. SBERT Sentence Embedding
    embeddings = embed_sentences(sentences, lang=lang, use_finetuned=use_finetuned)

    # 4. Compute Adaptive K
    kmeans_k, target_k = compute_k_adaptive(len(sentences), summary_length=length, enable_buffer=True)

    # 5. K-Means Clustering & Intrinsic Metric (Silhouette Score)
    indices, sents, embs, sil_score = kmeans_cluster(sentences, embeddings, kmeans_k)

    # 6. Post-filtering (Semantic Deduplication) with dynamic target_sents
    f_indices, f_sents, f_embs = filter_redundant(indices, sents, embs, target_sents=target_k)

    # 7. Original Sequence Reordering
    ordered_sents = reorder_by_original(f_indices, f_sents)

    # 8. Diversity Score Calculation
    div_score = diversity_score(f_embs)

    latency_ms = round((time.time() - start_time) * 1000, 2)
    original_count = len(sentences)
    summary_count = len(ordered_sents)
    compression_ratio = round((1.0 - (summary_count / max(1, original_count))) * 100, 1)

    return {
        'original_text': text,
        'summary_text': " ".join(ordered_sents),
        'summary_sentences': ordered_sents,
        'highlight_indices': f_indices,
        'stats': {
            'original_sentence_count': original_count,
            'summary_sentence_count': summary_count,
            'compression_ratio': compression_ratio,
            'latency_ms': latency_ms,
            'silhouette_score': round(sil_score, 4),
            'diversity_score': round(div_score, 4),
            'detected_language': lang.upper()
        }
    }
