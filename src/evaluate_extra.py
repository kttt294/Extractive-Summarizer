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
    from src.evaluate import run_lead3_baseline, run_textrank_baseline, run_sbert_no_kmeans_pipeline, run_sbert_pipeline, compute_bertscore_f1
except ImportError:
    from preprocess import preprocess_text
    from embedding import embed_sentences
    from summarizer import compute_k_adaptive, kmeans_cluster, filter_redundant, reorder_by_original, diversity_score
    from evaluate import run_lead3_baseline, run_textrank_baseline, run_sbert_no_kmeans_pipeline, run_sbert_pipeline, compute_bertscore_f1

def evaluate_extra_framework(dataset_name, subset, text_col, summary_col, sample_count=2000, lang='en', split='test'):
    divider = "=" * 126
    print(f"\nĐánh giá trên tập dữ liệu {dataset_name} ({subset}) - Số lượng: {sample_count}\n")
    
    if subset:
        ds = load_dataset(dataset_name, subset, split=split, trust_remote_code=True)
    else:
        ds = load_dataset(dataset_name, split=split, trust_remote_code=True)
    
    ds = ds.select(range(min(sample_count, len(ds))))
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    models_to_test = ['Lead-3', 'TextRank', 'Pretrained-SBERT-KMeans', 'SBERT-No-KMeans', 'FineTuned-SBERT-KMeans']
    results = {m: {'r1': [], 'r2': [], 'rl': [], 'bert': [], 'sil': [], 'div': [], 'comp': []} for m in models_to_test}
    
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
            results['Lead-3']['bert'].append(compute_bertscore_f1(lead3_summary, reference, lang=lang))
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
            results['TextRank']['bert'].append(compute_bertscore_f1(tr_summary, reference, lang=lang))
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
            results['Pretrained-SBERT-KMeans']['bert'].append(compute_bertscore_f1(pre_summary, reference, lang=lang))
            results['Pretrained-SBERT-KMeans']['sil'].append(sil_pre)
            results['Pretrained-SBERT-KMeans']['div'].append(div_pre)
            results['Pretrained-SBERT-KMeans']['comp'].append((1.0 - len(pre_summary.split()) / art_words) * 100)

            # 4. SBERT-No-KMeans
            nokm_summary, _, _, div_nokm = run_sbert_no_kmeans_pipeline(article, lang=lang, use_finetuned=True)
            s_nokm = scorer.score(reference, nokm_summary)
            results['SBERT-No-KMeans']['r1'].append(s_nokm['rouge1'].fmeasure)
            results['SBERT-No-KMeans']['r2'].append(s_nokm['rouge2'].fmeasure)
            results['SBERT-No-KMeans']['rl'].append(s_nokm['rougeL'].fmeasure)
            results['SBERT-No-KMeans']['bert'].append(compute_bertscore_f1(nokm_summary, reference, lang=lang))
            results['SBERT-No-KMeans']['div'].append(div_nokm)
            results['SBERT-No-KMeans']['comp'].append((1.0 - len(nokm_summary.split()) / art_words) * 100)

            # 5. Fine-Tuned SBERT + K-Means
            ft_summary, _, sil_ft, div_ft = run_sbert_pipeline(article, lang=lang, use_finetuned=True)
            s_ft = scorer.score(reference, ft_summary)
            results['FineTuned-SBERT-KMeans']['r1'].append(s_ft['rouge1'].fmeasure)
            results['FineTuned-SBERT-KMeans']['r2'].append(s_ft['rouge2'].fmeasure)
            results['FineTuned-SBERT-KMeans']['rl'].append(s_ft['rougeL'].fmeasure)
            results['FineTuned-SBERT-KMeans']['bert'].append(compute_bertscore_f1(ft_summary, reference, lang=lang))
            results['FineTuned-SBERT-KMeans']['sil'].append(sil_ft)
            results['FineTuned-SBERT-KMeans']['div'].append(div_ft)
            results['FineTuned-SBERT-KMeans']['comp'].append((1.0 - len(ft_summary.split()) / art_words) * 100)
        except Exception:
            continue
            
    non_clustering_models = {'Lead-3', 'TextRank', 'SBERT-No-KMeans'}

    print("\n" + divider)
    print(f"{'Mô hình':<26} | {'Silhouette':<10} | {'Diversity':<10} | {'Compress (%)':<12} | {'ROUGE-1 (%)':<12} | {'ROUGE-2 (%)':<12} | {'ROUGE-L (%)':<12} | {'BERTScore':<10}")
    print(divider)

    for m in models_to_test:
        div_m = np.mean(results[m]['div']) if results[m]['div'] else 0.0
        comp_m = np.mean(results[m]['comp']) if results[m]['comp'] else 0.0
        r1_m = np.mean(results[m]['r1']) * 100
        r2_m = np.mean(results[m]['r2']) * 100
        rl_m = np.mean(results[m]['rl']) * 100
        bert_m = np.mean(results[m]['bert']) if results[m]['bert'] else 0.0

        if m in non_clustering_models:
            sil_str = "N/A       "
        else:
            sil_m = np.mean(results[m]['sil']) if results[m]['sil'] else 0.0
            sil_str = f"{sil_m:<10.4f}"

        print(f"{m:<26} | {sil_str} | {div_m:<10.4f} | {comp_m:<12.2f} | {r1_m:<12.4f} | {r2_m:<12.4f} | {rl_m:<12.4f} | {bert_m:<10.4f}")

    print(divider + "\n\n")

def run_extra_evaluations(num_samples=2000):
    print(f"\nĐánh giá trên ngôn ngữ EN - Số lượng: {num_samples}\n")
    evaluate_extra_framework("ccdv/pubmed-summarization", "document", "article", "abstract", num_samples, lang='en', split='test')
    evaluate_extra_framework("dany0407/reddit_tifu_long", "", "documents", "tldr", num_samples, lang='en', split='train')

if __name__ == '__main__':
    run_extra_evaluations(2000)