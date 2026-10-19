import time
from threading import Lock

from src.preprocess import resolve_language, preprocess_text
from src.summarizer import (compute_k_adaptive, kmeans_cluster, filter_redundant,
                            reorder_by_original, diversity_score)
from backend.services.advanced_predictor import get_predictor, source_spans

_english_lock = Lock()


def process_summarization(text, user_lang='auto', length='medium', use_finetuned=True):
    start = time.perf_counter()
    lang = resolve_language(text, user_choice=user_lang)
    if lang == 'vi':
        sentences, indices, embeddings = get_predictor().summarize(text)
        summary = [sentences[i] for i in indices]
        selected_embs = [embeddings[i] for i in indices]
        silhouette = None
        method, version = 'SBERT + Logistic Regression', 'advanced-ml-v2-logistic-seed42'
    else:
        # Tiếng Anh kế thừa, không tuyên bố có kết quả v2 trên tiếng Anh.
        import src.config as config
        from src.embedding import get_sbert_model
        with _english_lock:
            config.set_language_config('en')
            pairs = preprocess_text(text, lang='en')
            if not pairs:
                raise ValueError('Văn bản không chứa câu hợp lệ để tóm tắt.')
            encoder = get_sbert_model('en', use_finetuned=False, strict=True)
            embeddings = encoder.encode([s for _, s in pairs], convert_to_numpy=True,
                                        normalize_embeddings=True, show_progress_bar=False)
            k, target = compute_k_adaptive(len(pairs), summary_length=length)
            ids, sents, embs, silhouette = kmeans_cluster(pairs, embeddings, k)
            ids, sents, selected_embs = filter_redundant(ids, sents, embs, target_sents=target)
            indices = sorted(ids)
            summary = reorder_by_original(ids, sents)
            sentences = [s for _, s in pairs]
        method, version = 'SBERT + K-Means', 'legacy-en'
    diversity = diversity_score(selected_embs) if len(selected_embs) >= 2 else None
    return dict(
        original_text=text, original_sentence_spans=source_spans(text, sentences),
        summary_text=' '.join(summary), summary_sentences=summary, highlight_indices=indices,
        stats=dict(original_sentence_count=len(sentences), summary_sentence_count=len(summary),
                   compression_ratio=round((1-len(summary)/len(sentences))*100, 1),
                   latency_ms=round((time.perf_counter()-start)*1000, 2),
                   silhouette_score=round(silhouette, 4) if silhouette is not None else None,
                   diversity_score=diversity, detected_language=lang.upper(),
                   method=method, model_version=version,
                   summary_policy='fixed-3' if lang == 'vi' else 'adaptive',
                   note=('Tiếng Việt: chọn tối đa 3 câu theo cấu hình thực nghiệm. '
                         'Văn bản ít hơn 3 câu hợp lệ sẽ giữ các câu hiện có.' if lang == 'vi'
                         else 'Tiếng Anh dùng phương pháp SBERT + K-Means kế thừa.')))
