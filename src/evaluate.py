import os
import warnings
import logging

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

try:
    from transformers import logging as tf_logging
    tf_logging.set_verbosity_error()
except Exception:
    pass

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sbert_score").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)

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
        # Fallback to custom TextRank for Vietnamese (since sumy lacks VN tokenizer)
        sentences = preprocess_text(text, lang=lang)
        if len(sentences) <= n_sentences:
            return [s for _, s in sentences]
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            import networkx as nx
            
            texts = [s for _, s in sentences]
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(texts)
            sim_matrix = cosine_similarity(tfidf_matrix)
            
            nx_graph = nx.from_numpy_array(sim_matrix)
            scores = nx.pagerank(nx_graph)
            
            ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(texts)), reverse=True)
            original_indices = {s: i for i, s in sentences}
            top_sents = [s for _, s in ranked_sentences[:n_sentences]]
            top_sents.sort(key=lambda x: original_indices[x])
            return top_sents
        except Exception:
            return [s for _, s in sentences[:n_sentences]]


def run_sbert_no_kmeans_pipeline(text: str, lang: str = 'en', use_finetuned: bool = False):
    """
    Ablation Study: Fine-Tuned SBERT + Direct Top-K (Không có K-Means)
    Lấy Top-K câu có khoảng cách Cosine gần nhất với Vector trung bình toàn bài báo
    """
    import src.config as config
    config.set_language_config(lang)
    
    sentences = preprocess_text(text, lang=lang)
    if len(sentences) == 0:
        return "", [], 0.0, 0.0

    embeddings = embed_sentences(sentences, lang=lang, use_finetuned=use_finetuned)
    _, target_k = compute_k_adaptive(len(sentences))

    # Tính Vector trung bình toàn bài báo (Document Centroid)
    doc_embedding = np.mean(embeddings, axis=0)

    # Tính Cosine Similarity của từng câu với Document Centroid
    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity(embeddings, doc_embedding.reshape(1, -1)).flatten()

    # Lấy Top-K câu có similarity cao nhất
    top_k_indices = np.argsort(sims)[-target_k:][::-1]
    top_k_indices = sorted(top_k_indices)
    
    selected_sents = [sentences[idx][1] for idx in top_k_indices]
    
    # Tính Diversity trên không gian vector cơ sở chuẩn (Neutral Reference Space)
    base_embs = embed_sentences([(i, s) for i, s in enumerate(selected_sents)], lang=lang, use_finetuned=False)
    div_score = diversity_score(base_embs)
    summary_text = " ".join(selected_sents)
    return summary_text, selected_sents, 0.0, div_score


def run_sbert_pipeline(text: str, lang: str = 'en', use_finetuned: bool = False):
    """
    Chạy toàn bộ Pipeline 2 Giai đoạn
    Tiền xử lý -> SBERT Embedding -> K thích ứng -> K-Means + Lọc trùng -> Sắp xếp lại thứ tự gốc
    """
    import src.config as config
    config.set_language_config(lang)
    
    sentences = preprocess_text(text, lang=lang)
    if len(sentences) == 0:
        return "", [], 0.0, 0.0

    embeddings = embed_sentences(sentences, lang=lang, use_finetuned=use_finetuned)
    kmeans_k, target_k = compute_k_adaptive(len(sentences))

    indices, sents, embs, sil_score = kmeans_cluster(sentences, embeddings, kmeans_k)
    f_indices, f_sents, f_embs = filter_redundant(indices, sents, embs, target_sents=target_k)
    ordered_sents = reorder_by_original(f_indices, f_sents)

    # Tính Diversity trên không gian vector cơ sở chuẩn (Neutral Reference Space) để đảm bảo tính công bằng
    base_embs = embed_sentences([(i, s) for i, s in enumerate(f_sents)], lang=lang, use_finetuned=False)
    div_score = diversity_score(base_embs)
    summary_text = " ".join(ordered_sents)

    return summary_text, ordered_sents, sil_score, div_score


import os
import warnings
import logging

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sbert_score").setLevel(logging.ERROR)


def compute_sbert_cosine_similarity(summary: str, reference: str, lang: str = 'en') -> float:
    """
    Tính điểm BERTScore / Semantic Similarity F1 thông qua Vector SBERT/BERT.
    Đo tương đồng ngữ nghĩa cấp độ vector giữa bản tóm tắt máy và bản tóm tắt chuẩn.
    """
    if not summary or not reference:
        return 0.0
    try:
        emb_sum = embed_sentences([(0, summary)], lang=lang, use_finetuned=False)
        emb_ref = embed_sentences([(0, reference)], lang=lang, use_finetuned=False)
        if len(emb_sum) == 0 or len(emb_ref) == 0:
            return 0.0
        from sklearn.metrics.pairwise import cosine_similarity
        score = float(cosine_similarity(emb_sum, emb_ref)[0][0])
        # Chuẩn hóa về dải [0, 1]
        return max(0.0, min(1.0, score))
    except Exception:
        return 0.0

# Cache toàn cục cho BERTScorer
_bert_scorer = None
_bert_scorer_lang = None

def compute_real_bertscore(summary: str, reference: str, lang: str = 'en') -> float:
    global _bert_scorer, _bert_scorer_lang
    try:
        from bert_score import BERTScorer
        if _bert_scorer is None or _bert_scorer_lang != lang:
            model_type = "bert-base-multilingual-cased" if lang == 'vi' else "roberta-large"
            _bert_scorer = BERTScorer(lang=lang, model_type=model_type)
            _bert_scorer_lang = lang
            
        P, R, F1 = _bert_scorer.score([summary], [reference])
        return float(F1.mean().item())
    except Exception:
        return 0.0

def evaluate_framework(lang: str = 'en', sample_count: int = 2000):
    divider = "=" * 98
    print(f"Đánh giá trên ngôn ngữ {lang.upper()} - Số lượng: {sample_count}")

    test_samples = load_evaluation_dataset(lang=lang, sample_count=sample_count, split='test')
    if not test_samples:
        print("Không tìm thấy dữ liệu thử nghiệm.")
        return

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    models_to_test = ['Lead-3', 'TextRank', 'Pretrained-SBERT-KMeans', 'SBERT-No-KMeans', 'FineTuned-SBERT-KMeans']
    results = {m: {'r1': [], 'r2': [], 'rl': [], 'sbert_cosine': [], 'real_bertscore': [], 'sil': [], 'div': [], 'comp': []} for m in models_to_test}

    for idx, sample in enumerate(test_samples):
        article = sample['article']
        reference = sample['highlights']
        sentences = preprocess_text(article, lang=lang)
        art_words = max(1, len(article.split()))

        # 1. Lead-3
        lead3_sents = run_lead3_baseline(sentences)
        lead3_summary = " ".join(lead3_sents)
        s_lead3 = scorer.score(reference, lead3_summary)
        results['Lead-3']['r1'].append(s_lead3['rouge1'].fmeasure)
        results['Lead-3']['r2'].append(s_lead3['rouge2'].fmeasure)
        results['Lead-3']['rl'].append(s_lead3['rougeL'].fmeasure)
        results['Lead-3']['sbert_cosine'].append(compute_sbert_cosine_similarity(lead3_summary, reference, lang=lang))
        results['Lead-3']['real_bertscore'].append(compute_real_bertscore(lead3_summary, reference, lang=lang))
        results['Lead-3']['comp'].append((1.0 - len(lead3_summary.split()) / art_words) * 100)
        # Tính Diversity công bằng cho Lead-3 bằng cách embed các câu đã chọn qua SBERT (Neutral Reference Space)
        if len(lead3_sents) >= 2:
            lead3_embs = embed_sentences([(i, s) for i, s in enumerate(lead3_sents)], lang=lang, use_finetuned=False)
            results['Lead-3']['div'].append(diversity_score(lead3_embs))
        else:
            results['Lead-3']['div'].append(0.0)

        # 2. TextRank
        tr_sents = run_textrank_baseline(article, n_sentences=3, lang=lang)
        tr_summary = " ".join(tr_sents)
        s_tr = scorer.score(reference, tr_summary)
        results['TextRank']['r1'].append(s_tr['rouge1'].fmeasure)
        results['TextRank']['r2'].append(s_tr['rouge2'].fmeasure)
        results['TextRank']['rl'].append(s_tr['rougeL'].fmeasure)
        results['TextRank']['sbert_cosine'].append(compute_sbert_cosine_similarity(tr_summary, reference, lang=lang))
        results['TextRank']['real_bertscore'].append(compute_real_bertscore(tr_summary, reference, lang=lang))
        results['TextRank']['comp'].append((1.0 - len(tr_summary.split()) / art_words) * 100)
        # Tính Diversity công bằng cho TextRank bằng cách embed các câu đã chọn qua SBERT (Neutral Reference Space)
        if len(tr_sents) >= 2:
            tr_embs = embed_sentences([(i, s) for i, s in enumerate(tr_sents)], lang=lang, use_finetuned=False)
            results['TextRank']['div'].append(diversity_score(tr_embs))
        else:
            results['TextRank']['div'].append(0.0)

        # 3. Pretrained SBERT + K-Means (Un-finetuned Baseline)
        pre_summary, _, sil_pre, div_pre = run_sbert_pipeline(article, lang=lang, use_finetuned=False)
        s_pre = scorer.score(reference, pre_summary)
        results['Pretrained-SBERT-KMeans']['r1'].append(s_pre['rouge1'].fmeasure)
        results['Pretrained-SBERT-KMeans']['r2'].append(s_pre['rouge2'].fmeasure)
        results['Pretrained-SBERT-KMeans']['rl'].append(s_pre['rougeL'].fmeasure)
        results['Pretrained-SBERT-KMeans']['sbert_cosine'].append(compute_sbert_cosine_similarity(pre_summary, reference, lang=lang))
        results['Pretrained-SBERT-KMeans']['real_bertscore'].append(compute_real_bertscore(pre_summary, reference, lang=lang))
        results['Pretrained-SBERT-KMeans']['sil'].append(sil_pre)
        results['Pretrained-SBERT-KMeans']['div'].append(div_pre)
        results['Pretrained-SBERT-KMeans']['comp'].append((1.0 - len(pre_summary.split()) / art_words) * 100)

        # 4. SBERT-No-KMeans (Ablation Study)
        nokm_summary, _, _, div_nokm = run_sbert_no_kmeans_pipeline(article, lang=lang, use_finetuned=True)
        s_nokm = scorer.score(reference, nokm_summary)
        results['SBERT-No-KMeans']['r1'].append(s_nokm['rouge1'].fmeasure)
        results['SBERT-No-KMeans']['r2'].append(s_nokm['rouge2'].fmeasure)
        results['SBERT-No-KMeans']['rl'].append(s_nokm['rougeL'].fmeasure)
        results['SBERT-No-KMeans']['sbert_cosine'].append(compute_sbert_cosine_similarity(nokm_summary, reference, lang=lang))
        results['SBERT-No-KMeans']['real_bertscore'].append(compute_real_bertscore(nokm_summary, reference, lang=lang))
        results['SBERT-No-KMeans']['div'].append(div_nokm)
        results['SBERT-No-KMeans']['comp'].append((1.0 - len(nokm_summary.split()) / art_words) * 100)

        # 5. Fine-Tuned SBERT + K-Means (Full Proposed Model)
        ft_summary, _, sil_ft, div_ft = run_sbert_pipeline(article, lang=lang, use_finetuned=True)
        s_ft = scorer.score(reference, ft_summary)
        results['FineTuned-SBERT-KMeans']['r1'].append(s_ft['rouge1'].fmeasure)
        results['FineTuned-SBERT-KMeans']['r2'].append(s_ft['rouge2'].fmeasure)
        results['FineTuned-SBERT-KMeans']['rl'].append(s_ft['rougeL'].fmeasure)
        results['FineTuned-SBERT-KMeans']['sbert_cosine'].append(compute_sbert_cosine_similarity(ft_summary, reference, lang=lang))
        results['FineTuned-SBERT-KMeans']['real_bertscore'].append(compute_real_bertscore(ft_summary, reference, lang=lang))
        results['FineTuned-SBERT-KMeans']['sil'].append(sil_ft)
        results['FineTuned-SBERT-KMeans']['div'].append(div_ft)
        results['FineTuned-SBERT-KMeans']['comp'].append((1.0 - len(ft_summary.split()) / art_words) * 100)

    # Các mô hình không sử dụng K-Means → Silhouette không áp dụng (N/A)
    non_clustering_models = {'Lead-3', 'TextRank', 'SBERT-No-KMeans'}

    # In Bảng Kết quả Đánh giá
    divider = "=" * 150
    print("\n" + divider)
    print(f"{'Mô hình':<26} | {'Silhouette':<10} | {'Diversity':<10} | {'Compress (%)':<12} | {'ROUGE-1 (%)':<18} | {'ROUGE-2 (%)':<18} | {'ROUGE-L (%)':<18} | {'SBERT Cosine':<14} | {'BERTScore':<14}")
    print(divider)

    for m in models_to_test:
        div_m = np.mean(results[m]['div']) if results[m]['div'] else 0.0
        comp_m = np.mean(results[m]['comp']) if results[m]['comp'] else 0.0
        r1_m = np.mean(results[m]['r1']) * 100
        r1_s = np.std(results[m]['r1']) * 100 if len(results[m]['r1']) > 1 else 0.0
        r2_m = np.mean(results[m]['r2']) * 100
        r2_s = np.std(results[m]['r2']) * 100 if len(results[m]['r2']) > 1 else 0.0
        rl_m = np.mean(results[m]['rl']) * 100
        rl_s = np.std(results[m]['rl']) * 100 if len(results[m]['rl']) > 1 else 0.0
        sbert_m = np.mean(results[m]['sbert_cosine']) if results[m]['sbert_cosine'] else 0.0
        sbert_s = np.std(results[m]['sbert_cosine']) if len(results[m]['sbert_cosine']) > 1 else 0.0

        bertscore_m = np.mean(results[m]['real_bertscore']) if results[m]['real_bertscore'] else 0.0
        bertscore_s = np.std(results[m]['real_bertscore']) if len(results[m]['real_bertscore']) > 1 else 0.0

        if m in non_clustering_models:
            sil_str = "N/A       "
        else:
            sil_m = np.mean(results[m]['sil']) if results[m]['sil'] else 0.0
            sil_str = f"{sil_m:<10.4f}"

        print(f"{m:<26} | {sil_str} | {div_m:<10.4f} | {comp_m:<12.2f} | {f'{r1_m:.2f} ± {r1_s:.2f}':<18} | {f'{r2_m:.2f} ± {r2_s:.2f}':<18} | {f'{rl_m:.2f} ± {rl_s:.2f}':<18} | {f'{sbert_m:.4f} ± {sbert_s:.4f}':<14} | {f'{bertscore_m:.4f} ± {bertscore_s:.4f}':<14}")

    print(divider)
    
    # Paired t-test for significance (ROUGE-1 and Real BERTScore)
    try:
        from scipy import stats
        if len(results['Lead-3']['r1']) > 1 and len(results['FineTuned-SBERT-KMeans']['r1']) > 1:
            _, p_value_r1 = stats.ttest_rel(results['Lead-3']['r1'], results['FineTuned-SBERT-KMeans']['r1'])
            _, p_value_bert = stats.ttest_rel(results['Lead-3']['real_bertscore'], results['FineTuned-SBERT-KMeans']['real_bertscore'])
            print(f"\n[Statistical Significance Test - Paired t-test (Lead-3 vs FineTuned-SBERT-KMeans)]")
            print(f"- ROUGE-1 p-value:      {p_value_r1:.4e} ({'Significant (p<0.05)' if p_value_r1 < 0.05 else 'Not Significant'})")
            print(f"- Real BERTScore p-value: {p_value_bert:.4e} ({'Significant (p<0.05)' if p_value_bert < 0.05 else 'Not Significant'})")
    except Exception as e:
        print(f"\nLỗi khi tính p-value: {e}")

    print("\n\n")


if __name__ == "__main__":
    evaluate_framework(lang='en', sample_count=2000)
    evaluate_framework(lang='vi', sample_count=2000)
