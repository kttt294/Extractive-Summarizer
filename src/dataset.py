import nltk
from typing import List, Tuple
from datasets import load_dataset
from rouge_score import rouge_scorer
from sentence_transformers import InputExample


def load_evaluation_dataset(lang: str = 'en', sample_count: int = 200):
    """
    Nạp dữ liệu thử nghiệm:
    - Tiếng Anh: CNN/DailyMail v3.0.0
    - Tiếng Việt: VietNews sample / Fallback
    """
    print(f"Đang nạp bộ dữ liệu thử nghiệm {lang.upper()} (số lượng={sample_count})...")
    if lang == 'en':
        ds = load_dataset("cnn_dailymail", "3.0.0", split="test")
        samples = ds.select(range(min(sample_count, len(ds))))
        return [
            {
                'id': idx,
                'article': sample['article'],
                'highlights': sample['highlights']
            }
            for idx, sample in enumerate(samples)
        ]
    else:
        try:
            ds = load_dataset("vietnews", split="test")
            samples = ds.select(range(min(sample_count, len(ds))))
            return [
                {
                    'id': idx,
                    'article': sample['article'],
                    'highlights': sample['abstract']
                }
                for idx, sample in enumerate(samples)
            ]
        except Exception:
            print("Chưa nạp trực tiếp được dataset VietNews. Đang dùng mẫu thử nghiệm.")
            return []


def generate_oracle_extractive_pairs(articles_data: List[dict], max_pairs: int = 2000) -> List[InputExample]:
    """
    Tự động sinh các cặp câu Oracle (Câu bài báo <-> Câu tóm tắt chuẩn)
    dùng điểm ROUGE-1 làm nhãn (Pull nếu >0.45, Push nếu <0.10).
    """
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    training_examples = []

    print(f"Đang tự động tạo các cặp câu Oracle cho Fine-Tuning (mục tiêu: {max_pairs} cặp)...")
    for sample in articles_data:
        if len(training_examples) >= max_pairs:
            break

        article_sents = nltk.sent_tokenize(sample['article'])
        summary_sents = nltk.sent_tokenize(sample['highlights'])

        for a_sent in article_sents[:8]:  # Tập trung vào các câu mở đầu
            for s_sent in summary_sents:
                score = float(scorer.score(s_sent, a_sent)['rouge1'].fmeasure)
                
                # Cặp PULL (label = 1.0) nếu độ tương đồng cao
                if score > 0.45:
                    training_examples.append(InputExample(texts=[a_sent, s_sent], label=1.0))
                # Cặp PUSH (label = 0.0) nếu không có tương đồng
                elif score < 0.10:
                    training_examples.append(InputExample(texts=[a_sent, s_sent], label=0.0))

                if len(training_examples) >= max_pairs:
                    break

    print(f"Đã khởi tạo thành công {len(training_examples)} cặp câu huấn luyện Oracle.")
    return training_examples
