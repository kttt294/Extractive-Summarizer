import numpy as np
from typing import List, Dict
from rouge_score import rouge_scorer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from src.preprocess import preprocess_text
from src.embedding import embed_sentences
from src.summarizer import compute_k_adaptive, kmeans_cluster, filter_redundant, reorder_by_original, diversity_score
from src.dataset import load_evaluation_dataset


def run_lead3_baseline(sentences: List[tuple]) -> List[str]:
    """Lead-3 Baseline: Lấy 3 câu đầu tiên của bài báo"""
    return [text for _, text in sentences[:3]]


def run_textrank_baseline(text: str, n_sentences: int = 3, lang: str = 'en') -> List[str]:
    """TextRank Baseline qua thư viện Sumy"""
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english" if lang == 'en' else "vietnamese"))
        summarizer = TextRankSummarizer()
        summary_sents = summarizer(parser.document, n_sentences)
        return [str(s) for s in summary_sents]
    except Exception:
        return [s for _, s in preprocess_text(text, lang=lang)[:n_sentences]]


def run_sbert_no_kmeans_pipeline(text: str, lang: str = 'en', use_finetuned: bool = False):
    """
    Ablation Study: Fine-Tuned SBERT + Direct Top-K (Không có K-Means)
    Lấy Top-K câu có khoảng cách Cosine gần nhất với Vector trung bình toàn bài báo
    """
    sentences = preprocess_text(text, lang=lang)
    if len(sentences) == 0:
        return "", [], 0.0, 0.0

    embeddings = embed_sentences(sentences, lang=lang, use_finetuned=use_finetuned)
    k = compute_k_adaptive(len(sentences))

    # Tính Vector trung bình toàn bài báo (Document Centroid)
    doc_embedding = np.mean(embeddings, axis=0)

    # Tính Cosine Similarity của từng câu với Document Centroid
    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity(embeddings, doc_embedding.reshape(1, -1)).flatten()

    # Lấy Top-K câu có similarity cao nhất
    top_k_indices = np.argsort(sims)[-k:][::-1]
    top_k_indices = sorted(top_k_indices)
    
    selected_sents = [sentences[idx][1] for idx in top_k_indices]
    selected_embs = embeddings[top_k_indices]
    
    div_score = diversity_score(selected_embs)
    summary_text = " ".join(selected_sents)
    return summary_text, selected_sents, 0.0, div_score


def run_sbert_pipeline(text: str, lang: str = 'en', use_finetuned: bool = False):
    """
    Chạy toàn bộ Pipeline 2 Giai đoạn
    Tiền xử lý -> SBERT Embedding -> K thích ứng -> K-Means + Lọc trùng -> Sắp xếp lại thứ tự gốc
    """
    sentences = preprocess_text(text, lang=lang)
    if len(sentences) == 0:
        return "", [], 0.0, 0.0

    embeddings = embed_sentences(sentences, lang=lang, use_finetuned=use_finetuned)
    k = compute_k_adaptive(len(sentences))

    indices, sents, embs, sil_score = kmeans_cluster(sentences, embeddings, k)
    f_indices, f_sents, f_embs = filter_redundant(indices, sents, embs)
    ordered_sents = reorder_by_original(f_indices, f_sents)

    div_score = diversity_score(f_embs)
    summary_text = " ".join(ordered_sents)

    return summary_text, ordered_sents, sil_score, div_score


def compute_bertscore_f1(summary: str, reference: str, lang: str = 'en') -> float:
    """Tính điểm BERTScore F1 thông qua BERTScore library hoặc Cosine Similarity Vector SBERT/BERT"""
    if not summary or not reference:
        return 0.0
    try:
        from bert_score import score
        P, R, F1 = score([summary], [reference], lang=lang, verbose=False)
        return float(F1.mean().item())
    except Exception:
        emb_sum = embed_sentences([(0, summary)], lang=lang, use_finetuned=True)
        emb_ref = embed_sentences([(0, reference)], lang=lang, use_finetuned=True)
        if len(emb_sum) == 0 or len(emb_ref) == 0:
            return 0.0
        from sklearn.metrics.pairwise import cosine_similarity
        return float(cosine_similarity(emb_sum, emb_ref)[0][0])


def evaluate_framework(lang: str = 'en', sample_count: int = 50):
    print(f"Đánh giá với ngôn ngữ {lang.upper()} - Số lượng: {sample_count})")

    test_samples = load_evaluation_dataset(lang=lang, sample_count=sample_count, split='test')
    if not test_samples:
        print("Không tìm thấy dữ liệu thử nghiệm.")
        return

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    models_to_test = ['Lead-3', 'TextRank', 'Pretrained-SBERT-KMeans', 'SBERT-No-KMeans', 'FineTuned-SBERT-KMeans']
    results = {m: {'r1': [], 'r2': [], 'rl': [], 'bert': [], 'sil': [], 'div': [], 'comp': []} for m in models_to_test}

    for idx, sample in enumerate(test_samples):
        article = sample['article']
        reference = sample['highlights']
        sentences = preprocess_text(article, lang=lang)
        art_words = max(1, len(article.split()))

        # 1. Lead-3
        lead3_summary = " ".join(run_lead3_baseline(sentences))
        s_lead3 = scorer.score(reference, lead3_summary)
        results['Lead-3']['r1'].append(s_lead3['rouge1'].fmeasure)
        results['Lead-3']['r2'].append(s_lead3['rouge2'].fmeasure)
        results['Lead-3']['rl'].append(s_lead3['rougeL'].fmeasure)
        results['Lead-3']['bert'].append(compute_bertscore_f1(lead3_summary, reference, lang=lang))
        results['Lead-3']['comp'].append(len(lead3_summary.split()) / art_words * 100)

        # 2. TextRank
        tr_summary = " ".join(run_textrank_baseline(article, n_sentences=3, lang=lang))
        s_tr = scorer.score(reference, tr_summary)
        results['TextRank']['r1'].append(s_tr['rouge1'].fmeasure)
        results['TextRank']['r2'].append(s_tr['rouge2'].fmeasure)
        results['TextRank']['rl'].append(s_tr['rougeL'].fmeasure)
        results['TextRank']['bert'].append(compute_bertscore_f1(tr_summary, reference, lang=lang))
        results['TextRank']['comp'].append(len(tr_summary.split()) / art_words * 100)

        # 3. Pretrained SBERT + K-Means (Un-finetuned Baseline)
        pre_summary, _, sil_pre, div_pre = run_sbert_pipeline(article, lang=lang, use_finetuned=False)
        s_pre = scorer.score(reference, pre_summary)
        results['Pretrained-SBERT-KMeans']['r1'].append(s_pre['rouge1'].fmeasure)
        results['Pretrained-SBERT-KMeans']['r2'].append(s_pre['rouge2'].fmeasure)
        results['Pretrained-SBERT-KMeans']['rl'].append(s_pre['rougeL'].fmeasure)
        results['Pretrained-SBERT-KMeans']['bert'].append(compute_bertscore_f1(pre_summary, reference, lang=lang))
        results['Pretrained-SBERT-KMeans']['sil'].append(sil_pre)
        results['Pretrained-SBERT-KMeans']['div'].append(div_pre)
        results['Pretrained-SBERT-KMeans']['comp'].append(len(pre_summary.split()) / art_words * 100)

        # 4. SBERT-No-KMeans (Ablation Study)
        nokm_summary, _, _, div_nokm = run_sbert_no_kmeans_pipeline(article, lang=lang, use_finetuned=True)
        s_nokm = scorer.score(reference, nokm_summary)
        results['SBERT-No-KMeans']['r1'].append(s_nokm['rouge1'].fmeasure)
        results['SBERT-No-KMeans']['r2'].append(s_nokm['rouge2'].fmeasure)
        results['SBERT-No-KMeans']['rl'].append(s_nokm['rougeL'].fmeasure)
        results['SBERT-No-KMeans']['bert'].append(compute_bertscore_f1(nokm_summary, reference, lang=lang))
        results['SBERT-No-KMeans']['div'].append(div_nokm)
        results['SBERT-No-KMeans']['comp'].append(len(nokm_summary.split()) / art_words * 100)

        # 5. Fine-Tuned SBERT + K-Means (Full Proposed Model)
        ft_summary, _, sil_ft, div_ft = run_sbert_pipeline(article, lang=lang, use_finetuned=True)
        s_ft = scorer.score(reference, ft_summary)
        results['FineTuned-SBERT-KMeans']['r1'].append(s_ft['rouge1'].fmeasure)
        results['FineTuned-SBERT-KMeans']['r2'].append(s_ft['rouge2'].fmeasure)
        results['FineTuned-SBERT-KMeans']['rl'].append(s_ft['rougeL'].fmeasure)
        results['FineTuned-SBERT-KMeans']['bert'].append(compute_bertscore_f1(ft_summary, reference, lang=lang))
        results['FineTuned-SBERT-KMeans']['sil'].append(sil_ft)
        results['FineTuned-SBERT-KMeans']['div'].append(div_ft)
        results['FineTuned-SBERT-KMeans']['comp'].append(len(ft_summary.split()) / art_words * 100)

    # In Bảng Kết quả Đánh giá
    divider = "=" * 126
    print("\n" + divider)
    print(f"{'Mô hình':<26} | {'Silhouette':<10} | {'Diversity':<10} | {'Compress (%)':<12} | {'ROUGE-1 (%)':<12} | {'ROUGE-2 (%)':<12} | {'ROUGE-L (%)':<12} | {'BERTScore':<10}")
    print(divider)

    for m in models_to_test:
        sil_m = np.mean(results[m]['sil']) if results[m]['sil'] else 0.0
        div_m = np.mean(results[m]['div']) if results[m]['div'] else 0.0
        comp_m = np.mean(results[m]['comp']) if results[m]['comp'] else 0.0
        r1_m = np.mean(results[m]['r1']) * 100
        r2_m = np.mean(results[m]['r2']) * 100
        rl_m = np.mean(results[m]['rl']) * 100
        bert_m = np.mean(results[m]['bert']) if results[m]['bert'] else 0.0

        print(f"{m:<26} | {sil_m:<10.4f} | {div_m:<10.4f} | {comp_m:<12.2f} | {r1_m:<12.4f} | {r2_m:<12.4f} | {rl_m:<12.4f} | {bert_m:<10.4f}")

    print(divider + "\n\n")


if __name__ == "__main__":
    evaluate_framework(lang='en', sample_count=50)
    evaluate_framework(lang='vi', sample_count=50)
