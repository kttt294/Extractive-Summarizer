import sys
import os
import warnings
import logging
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from rouge_score import rouge_scorer

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

try:
    from src.preprocess import preprocess_text
    from src.embedding import embed_sentences
    from src.summarizer import compute_k_adaptive, kmeans_cluster, filter_redundant, reorder_by_original, diversity_score
    from src.evaluate import run_lead3_baseline, run_textrank_baseline, run_sbert_no_kmeans_pipeline, run_sbert_pipeline, compute_sbert_cosine_similarity
except ImportError:
    from preprocess import preprocess_text
    from embedding import embed_sentences
    from summarizer import compute_k_adaptive, kmeans_cluster, filter_redundant, reorder_by_original, diversity_score
    from evaluate import run_lead3_baseline, run_textrank_baseline, run_sbert_no_kmeans_pipeline, run_sbert_pipeline, compute_sbert_cosine_similarity

def evaluate_extra_framework(dataset_name, subset, text_col, summary_col, sample_count=2000, lang='en', split='test'):
    divider = "=" * 150
    print(f"\nĐánh giá trên tập dữ liệu {dataset_name} ({subset}) - Số lượng: {sample_count}\n")
    
    if subset:
        ds = load_dataset(dataset_name, subset, split=split, trust_remote_code=True)
    else:
        ds = load_dataset(dataset_name, split=split, trust_remote_code=True)
    
    ds = ds.select(range(min(sample_count, len(ds))))
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    models_to_test = ['Lead-3', 'TextRank', 'Pretrained-SBERT-KMeans', 'SBERT-No-KMeans', 'FineTuned-SBERT-KMeans', 'FineTuned-Neutral-Lambda']
    results = {m: {'r1': [], 'r2': [], 'rl': [], 'sbert_cosine': [], 'sil': [], 'div': [], 'comp': []} for m in models_to_test}
    
    for item in tqdm(ds, desc=f"Đang xử lý {dataset_name}"):
        article = item[text_col]
        reference = item[summary_col]
        
        if not article or len(article.split()) < 20:
            continue
            
        sentences = preprocess_text(article, lang=lang)
        if len(sentences) == 0:
            continue
            
        art_words = max(1, len(article.split()))
        
        try:
            # 1. Lead-3
            lead3_sents = run_lead3_baseline(sentences)
            lead3_summary = " ".join(lead3_sents)
            s_lead3 = scorer.score(reference, lead3_summary)
            results['Lead-3']['r1'].append(s_lead3['rouge1'].fmeasure)
            results['Lead-3']['r2'].append(s_lead3['rouge2'].fmeasure)
            results['Lead-3']['rl'].append(s_lead3['rougeL'].fmeasure)
            results['Lead-3']['sbert_cosine'].append(compute_sbert_cosine_similarity(lead3_summary, reference, lang=lang))
            results['Lead-3']['comp'].append((1.0 - len(lead3_summary.split()) / art_words) * 100)
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
            results['TextRank']['comp'].append((1.0 - len(tr_summary.split()) / art_words) * 100)
            if len(tr_sents) >= 2:
                tr_embs = embed_sentences([(i, s) for i, s in enumerate(tr_sents)], lang=lang, use_finetuned=False)
                results['TextRank']['div'].append(diversity_score(tr_embs))
            else:
                results['TextRank']['div'].append(0.0)

            # 3. Pretrained SBERT + K-Means
            pre_summary, _, sil_pre, div_pre = run_sbert_pipeline(article, lang=lang, use_finetuned=False)
            s_pre = scorer.score(reference, pre_summary)
            results['Pretrained-SBERT-KMeans']['r1'].append(s_pre['rouge1'].fmeasure)
            results['Pretrained-SBERT-KMeans']['r2'].append(s_pre['rouge2'].fmeasure)
            results['Pretrained-SBERT-KMeans']['rl'].append(s_pre['rougeL'].fmeasure)
            results['Pretrained-SBERT-KMeans']['sbert_cosine'].append(compute_sbert_cosine_similarity(pre_summary, reference, lang=lang))
            results['Pretrained-SBERT-KMeans']['sil'].append(sil_pre)
            results['Pretrained-SBERT-KMeans']['div'].append(div_pre)
            results['Pretrained-SBERT-KMeans']['comp'].append((1.0 - len(pre_summary.split()) / art_words) * 100)

            # 4. SBERT-No-KMeans
            nokm_summary, _, _, div_nokm = run_sbert_no_kmeans_pipeline(article, lang=lang, use_finetuned=True)
            s_nokm = scorer.score(reference, nokm_summary)
            results['SBERT-No-KMeans']['r1'].append(s_nokm['rouge1'].fmeasure)
            results['SBERT-No-KMeans']['r2'].append(s_nokm['rouge2'].fmeasure)
            results['SBERT-No-KMeans']['rl'].append(s_nokm['rougeL'].fmeasure)
            results['SBERT-No-KMeans']['sbert_cosine'].append(compute_sbert_cosine_similarity(nokm_summary, reference, lang=lang))
            results['SBERT-No-KMeans']['div'].append(div_nokm)
            results['SBERT-No-KMeans']['comp'].append((1.0 - len(nokm_summary.split()) / art_words) * 100)

            # 5. Fine-Tuned SBERT + K-Means
            ft_summary, _, sil_ft, div_ft = run_sbert_pipeline(article, lang=lang, use_finetuned=True)
            s_ft = scorer.score(reference, ft_summary)
            results['FineTuned-SBERT-KMeans']['r1'].append(s_ft['rouge1'].fmeasure)
            results['FineTuned-SBERT-KMeans']['r2'].append(s_ft['rouge2'].fmeasure)
            results['FineTuned-SBERT-KMeans']['rl'].append(s_ft['rougeL'].fmeasure)
            results['FineTuned-SBERT-KMeans']['sbert_cosine'].append(compute_sbert_cosine_similarity(ft_summary, reference, lang=lang))
            results['FineTuned-SBERT-KMeans']['sil'].append(sil_ft)
            results['FineTuned-SBERT-KMeans']['div'].append(div_ft)
            results['FineTuned-SBERT-KMeans']['comp'].append((1.0 - len(ft_summary.split()) / art_words) * 100)

            # 6. Fine-Tuned SBERT + K-Means (Neutral Lambda = 0.0, No Length Filter)
            from src import config
            lang_dict = config.OPTIMAL_HYPERPARAMS_EN if lang == 'en' else config.OPTIMAL_HYPERPARAMS_VI
            
            original_lambda = lang_dict.get('lambda', 0.35)
            original_min = lang_dict.get('min_words', 4)
            original_max = lang_dict.get('max_words', 90)
            
            # Khởi tạo không gian trung lập: Tắt thiên vị vị trí và Tắt bộ lọc độ dài
            try:
                lang_dict['lambda'] = 0.0
                lang_dict['min_words'] = 1
                lang_dict['max_words'] = 9999
                
                ft_neu_summary, _, sil_ft_neu, div_ft_neu = run_sbert_pipeline(article, lang=lang, use_finetuned=True)
            finally:
                # Trả lại tham số gốc cho hệ thống báo chí dù có lỗi hay không
                lang_dict['lambda'] = original_lambda
                lang_dict['min_words'] = original_min
                lang_dict['max_words'] = original_max
            
            s_ft_neu = scorer.score(reference, ft_neu_summary)
            results['FineTuned-Neutral-Lambda']['r1'].append(s_ft_neu['rouge1'].fmeasure)
            results['FineTuned-Neutral-Lambda']['r2'].append(s_ft_neu['rouge2'].fmeasure)
            results['FineTuned-Neutral-Lambda']['rl'].append(s_ft_neu['rougeL'].fmeasure)
            results['FineTuned-Neutral-Lambda']['sbert_cosine'].append(compute_sbert_cosine_similarity(ft_neu_summary, reference, lang=lang))
            results['FineTuned-Neutral-Lambda']['sil'].append(sil_ft_neu)
            results['FineTuned-Neutral-Lambda']['div'].append(div_ft_neu)
            results['FineTuned-Neutral-Lambda']['comp'].append((1.0 - len(ft_neu_summary.split()) / art_words) * 100)
        except Exception as e:
            print(f"Lỗi bài báo: {e}")
            continue
            
    non_clustering_models = {'Lead-3', 'TextRank', 'SBERT-No-KMeans'}

    print("\n" + divider)
    print(f"{'Mô hình':<26} | {'Silhouette':<10} | {'Diversity':<10} | {'Compress (%)':<12} | {'ROUGE-1 (%)':<18} | {'ROUGE-2 (%)':<18} | {'ROUGE-L (%)':<18} | {'SBERT Cosine':<14}")
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

        if m in non_clustering_models:
            sil_str = "N/A       "
        else:
            sil_m = np.mean(results[m]['sil']) if results[m]['sil'] else 0.0
            sil_str = f"{sil_m:<10.4f}"

        print(f"{m:<26} | {sil_str} | {div_m:<10.4f} | {comp_m:<12.2f} | {f'{r1_m:.2f} ± {r1_s:.2f}':<18} | {f'{r2_m:.2f} ± {r2_s:.2f}':<18} | {f'{rl_m:.2f} ± {rl_s:.2f}':<18} | {f'{sbert_m:.4f} ± {sbert_s:.4f}':<14}")

    print(divider + "\n\n")

def run_extra_evaluations(num_samples=500):
    print(f"\nĐánh giá trên ngôn ngữ EN - Số lượng: {num_samples}\n")
    evaluate_extra_framework("ccdv/pubmed-summarization", "document", "article", "abstract", num_samples, lang='en', split='test')
    # Lưu ý: Tập dany0407/reddit_tifu_long không có tập 'test' nên lấy mẫu từ 'train'
    # Vì hệ thống không train trên bộ dữ liệu này (chỉ train trên VietNews), nên việc dùng 'train' để test là hợp lệ và không bị rò rỉ dữ liệu
    evaluate_extra_framework("dany0407/reddit_tifu_long", "", "documents", "tldr", num_samples, lang='en', split='train')

if __name__ == '__main__':
    run_extra_evaluations(500)