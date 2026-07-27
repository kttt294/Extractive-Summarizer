import os
import sys
import numpy as np
from rouge_score import rouge_scorer
import warnings
import logging

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

from src.dataset import load_evaluation_dataset
from src.preprocess import preprocess_text
from src.embedding import embed_sentences
import src.config as config
from src.summarizer import compute_k_adaptive, kmeans_cluster, filter_redundant, reorder_by_original

def run_pipeline_with_hyperparams(text, lang, use_finetuned, alpha, theta, pos_lambda):
    # Ghi đè tham số cục bộ
    config.OPTIMAL_HYPERPARAMS['alpha'] = alpha
    config.OPTIMAL_HYPERPARAMS['theta'] = theta
    config.OPTIMAL_HYPERPARAMS['lambda'] = pos_lambda

    sentences = preprocess_text(text, lang=lang)
    if len(sentences) == 0:
        return ""

    embeddings = embed_sentences(sentences, lang=lang, use_finetuned=use_finetuned)
    kmeans_k, target_k = compute_k_adaptive(len(sentences))

    indices, sents, embs, _ = kmeans_cluster(sentences, embeddings, kmeans_k)
    f_indices, f_sents, _ = filter_redundant(indices, sents, embs, threshold=theta, target_sents=target_k)
    ordered_sents = reorder_by_original(f_indices, f_sents)

    return " ".join(ordered_sents)

def grid_search(lang='vi', sample_count=20, dataset_name=None):
    if dataset_name:
        print(f"Bắt đầu tinh chỉnh siêu tham số trên tập dữ liệu ngoại lai: {dataset_name} với {sample_count} mẫu...")
        samples = load_evaluation_dataset(lang=lang, sample_count=sample_count, split='test', custom_dataset=dataset_name)
    else:
        print(f"Bắt đầu tinh chỉnh siêu tham số trên tập tin tức {lang.upper()} (VietNews/CNN) với {sample_count} mẫu...")
        samples = load_evaluation_dataset(lang=lang, sample_count=sample_count, split='test')
    
    alphas = [0.05, 0.10, 0.15, 0.20, 0.25]
    thetas = [0.65, 0.70, 0.75, 0.80, 0.85]
    lambdas = [0.20, 0.35, 0.50]
    
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    
    results = {}
    
    for pos_lambda in lambdas:
        for alpha in alphas:
            for theta in thetas:
                print(f"Đang chạy cấu hình: Lambda = {pos_lambda:.2f}, Alpha = {alpha:.2f}, Theta = {theta:.2f}")
                rouge1_scores = []
                for sample in samples:
                    article = sample['article']
                    reference = sample['highlights']
                    summary = run_pipeline_with_hyperparams(article, lang, True, alpha, theta, pos_lambda)
                if summary:
                    score = scorer.score(reference, summary)
                    rouge1_scores.append(score['rouge1'].fmeasure)
            
                mean_rouge1 = np.mean(rouge1_scores) if rouge1_scores else 0.0
                results[(pos_lambda, alpha, theta)] = mean_rouge1
                print(f"  -> ROUGE-1: {mean_rouge1*100:.2f}%")

    print("\n" + "="*60)
    print("BẢNG KẾT QUẢ GRID SEARCH (ROUGE-1 %)")
    print("="*60)
    
    for pos_lambda in lambdas:
        print(f"\n[ Lambda = {pos_lambda:.2f} ]")
        print(f"{'Alpha \\ Theta':<15} | " + " | ".join([f"{t:.2f}" for t in thetas]))
        print("-" * 60)
        for a in alphas:
            row = [f"{a:<15.2f}"]
            for t in thetas:
                score = results[(pos_lambda, a, t)] * 100
                row.append(f"{score:>4.2f}")
            print(" | ".join(row))

    best_params = max(results, key=results.get)
    print("\n" + "="*60)
    print(f"CẤU HÌNH TỐI ƯU NHẤT: Lambda = {best_params[0]:.2f}, Alpha = {best_params[1]:.2f}, Theta = {best_params[2]:.2f}")
    print(f"ROUGE-1 CAO NHẤT: {results[best_params]*100:.2f}%")

if __name__ == "__main__":
    grid_search(lang='vi', sample_count=20)
