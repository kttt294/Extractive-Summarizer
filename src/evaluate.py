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
    """Lead-3 Baseline: Lấy 3 câu đầu tiên của bài báo."""
    return [text for _, text in sentences[:3]]


def run_textrank_baseline(text: str, n_sentences: int = 3, lang: str = 'en') -> List[str]:
    """TextRank Baseline qua thư viện Sumy."""
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english" if lang == 'en' else "vietnamese"))
        summarizer = TextRankSummarizer()
        summary_sents = summarizer(parser.document, n_sentences)
        return [str(s) for s in summary_sents]
    except Exception:
        return [s for _, s in preprocess_text(text, lang=lang)[:n_sentences]]


def run_sbert_pipeline(text: str, lang: str = 'en', use_finetuned: bool = False):
    """
    Chạy toàn bộ Pipeline Tóm tắt Trích xuất:
    Tiền xử lý -> SBERT Embedding -> K thích ứng -> K-Means + Lọc trùng -> Sắp xếp lại thứ tự gốc.
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


def evaluate_framework(lang: str = 'en', sample_count: int = 50):
    """
    Chạy Khung Đánh giá Kép (Intrinsic + Extrinsic) so sánh:
    1. Lead-3 Baseline
    2. TextRank Baseline
    3. Pretrained SBERT
    4. Fine-Tuned SBERT
    """
    print(f"\n==========================================")
    print(f"  CHẠY KHUNG ĐÁNH GIÁ KÉP (NGÔN NGỮ {lang.upper()})")
    print(f"==========================================")

    test_samples = load_evaluation_dataset(lang=lang, sample_count=sample_count)
    if not test_samples:
        print("Không tìm thấy dữ liệu thử nghiệm.")
        return

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    models_to_test = ['Lead-3', 'TextRank', 'Pretrained-SBERT', 'FineTuned-SBERT']
    results = {m: {'r1': [], 'r2': [], 'rl': [], 'sil': [], 'div': []} for m in models_to_test}

    for idx, sample in enumerate(test_samples):
        article = sample['article']
        reference = sample['highlights']
        sentences = preprocess_text(article, lang=lang)

        # 1. Lead-3
        lead3_summary = " ".join(run_lead3_baseline(sentences))
        s_lead3 = scorer.score(reference, lead3_summary)
        results['Lead-3']['r1'].append(s_lead3['rouge1'].fmeasure)
        results['Lead-3']['r2'].append(s_lead3['rouge2'].fmeasure)
        results['Lead-3']['rl'].append(s_lead3['rougeL'].fmeasure)

        # 2. TextRank
        tr_summary = " ".join(run_textrank_baseline(article, n_sentences=3, lang=lang))
        s_tr = scorer.score(reference, tr_summary)
        results['TextRank']['r1'].append(s_tr['rouge1'].fmeasure)
        results['TextRank']['r2'].append(s_tr['rouge2'].fmeasure)
        results['TextRank']['rl'].append(s_tr['rougeL'].fmeasure)

        # 3. Pretrained SBERT
        p_summary, _, sil_p, div_p = run_sbert_pipeline(article, lang=lang, use_finetuned=False)
        s_p = scorer.score(reference, p_summary)
        results['Pretrained-SBERT']['r1'].append(s_p['rouge1'].fmeasure)
        results['Pretrained-SBERT']['r2'].append(s_p['rouge2'].fmeasure)
        results['Pretrained-SBERT']['rl'].append(s_p['rougeL'].fmeasure)
        results['Pretrained-SBERT']['sil'].append(sil_p)
        results['Pretrained-SBERT']['div'].append(div_p)

        # 4. Fine-Tuned SBERT
        ft_summary, _, sil_ft, div_ft = run_sbert_pipeline(article, lang=lang, use_finetuned=True)
        s_ft = scorer.score(reference, ft_summary)
        results['FineTuned-SBERT']['r1'].append(s_ft['rouge1'].fmeasure)
        results['FineTuned-SBERT']['r2'].append(s_ft['rouge2'].fmeasure)
        results['FineTuned-SBERT']['rl'].append(s_ft['rougeL'].fmeasure)
        results['FineTuned-SBERT']['sil'].append(sil_ft)
        results['FineTuned-SBERT']['div'].append(div_ft)

    # In Bảng Kết quả Đánh giá
    print("\n" + "=" * 90)
    print(f"{'Mô hình':<20} | {'Silhouette':<10} | {'Diversity':<10} | {'ROUGE-1':<10} | {'ROUGE-2':<10} | {'ROUGE-L':<10}")
    print("=" * 90)

    for m in models_to_test:
        sil_m = np.mean(results[m]['sil']) if results[m]['sil'] else 0.0
        div_m = np.mean(results[m]['div']) if results[m]['div'] else 0.0
        r1_m = np.mean(results[m]['r1'])
        r2_m = np.mean(results[m]['r2'])
        rl_m = np.mean(results[m]['rl'])

        print(f"{m:<20} | {sil_m:<10.4f} | {div_m:<10.4f} | {r1_m:<10.4f} | {r2_m:<10.4f} | {rl_m:<10.4f}")

    print("=" * 90 + "\n")


if __name__ == "__main__":
    evaluate_framework(lang='en', sample_count=20)
