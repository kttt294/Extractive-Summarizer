import warnings
import logging
import datasets
import nltk
from typing import List, Tuple
from datasets import load_dataset
from rouge_score import rouge_scorer
from sentence_transformers import InputExample

# Tắt toàn bộ cảnh báo dư thừa của Hugging Face Datasets & Warnings
warnings.filterwarnings("ignore")
datasets.logging.set_verbosity_error()
logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("datasets.load").setLevel(logging.ERROR)
logging.getLogger("datasets.builder").setLevel(logging.ERROR)

# Đảm bảo punkt và punkt_tab được tải sẵn cho Natural Language Toolkit (NLTK)
for resource in ['punkt', 'punkt_tab']:
    try:
        nltk.data.find(f'tokenizers/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)


def apply_quality_stratified_sampling(ds, sample_count: int, article_key: str = 'article', summary_key: str = 'highlights') -> List[dict]:
    """
    Chiến thuật Lấy mẫu Phân tầng & Lọc Chất lượng (Quality-Aware Stratified Sampling):
    1. Lọc chất lượng: Article >= 8 câu, Summary >= 2 câu.
    2. Lọc mật độ nén: Summary words / Article words <= 0.40 (Loại bỏ rác format).
    3. Phân tầng độ dài (Stratified Length-based Sampling):
       - Tầng Ngắn (8 - 15 câu): 33%
       - Tầng Trung bình (15 - 30 câu): 34%
       - Tầng Dài (> 30 câu): 33%
    """
    tier_short, tier_medium, tier_long = [], [], []
    target_per_tier = max(1, sample_count // 3)

    for idx, sample in enumerate(ds):
        art = sample.get(article_key) or sample.get('text', '')
        summ = sample.get(summary_key) or sample.get('abstract') or sample.get('summary', '')

        if not art or not summ:
            continue

        art_sents = [s.strip() for s in art.split('.') if len(s.strip()) > 5]
        summ_sents = [s.strip() for s in summ.split('.') if len(s.strip()) > 5]

        # 1. Lọc chất lượng văn bản
        if len(art_sents) < 8 or len(summ_sents) < 2:
            continue

        # 2. Lọc mật độ nén thông tin
        art_w = len(art.split())
        summ_w = len(summ.split())
        if art_w == 0 or (summ_w / art_w) > 0.40:
            continue

        item = {'id': idx, 'article': art, 'highlights': summ}
        num_sents = len(art_sents)

        # 3. Phân tầng theo độ dài bài báo
        if num_sents <= 15 and len(tier_short) < target_per_tier:
            tier_short.append(item)
        elif 15 < num_sents <= 30 and len(tier_medium) < target_per_tier:
            tier_medium.append(item)
        elif num_sents > 30 and len(tier_long) < target_per_tier:
            tier_long.append(item)

        if len(tier_short) >= target_per_tier and len(tier_medium) >= target_per_tier and len(tier_long) >= target_per_tier:
            break

    combined = tier_short + tier_medium + tier_long
    # Bổ sung nếu tập dữ liệu giới hạn
    if len(combined) < sample_count:
        existing_ids = {x['id'] for x in combined}
        for idx, sample in enumerate(ds):
            if len(combined) >= sample_count:
                break
            if idx in existing_ids:
                continue
            art = sample.get(article_key) or sample.get('text', '')
            summ = sample.get(summary_key) or sample.get('abstract') or sample.get('summary', '')
            if art and summ:
                combined.append({'id': idx, 'article': art, 'highlights': summ})

    return combined[:sample_count]


def load_evaluation_dataset(lang: str = 'en', sample_count: int = 200, split: str = 'test'):
    """
    Nạp dữ liệu chuẩn từ Hugging Face Hub bằng Chiến thuật Lấy mẫu Phân tầng & Lọc Chất lượng.
    Tham số split: 'train' (cho Fine-Tuning) hoặc 'test' (cho Đánh giá độc lập).
    """
    previous_verbosity = datasets.logging.get_verbosity()
    datasets.logging.set_verbosity_error()
    
    if lang == 'en':
        ds = None
        for dataset_name in ["cnn_dailymail", "abisee/cnn_dailymail"]:
            try:
                ds = load_dataset(dataset_name, "3.0.0", split=split)
                if ds is not None:
                    break
            except Exception:
                pass

        if ds is not None:
            samples = apply_quality_stratified_sampling(ds, sample_count=sample_count, article_key='article', summary_key='highlights')
            datasets.logging.set_verbosity(previous_verbosity)
            return samples
        else:
            datasets.logging.set_verbosity(previous_verbosity)
            raise RuntimeError(f"LỖI: Không thể nạp tập dữ liệu Tiếng Anh (CNN/DailyMail) từ Hugging Face Hub.")

    else:
        ds = None
        for dataset_name in ["nam194/vietnews", "truongpdd/vietnews-dataset"]:
            try:
                ds = load_dataset(dataset_name, split=split)
                if ds is not None:
                    break
            except Exception:
                pass

        if ds is not None:
            samples = apply_quality_stratified_sampling(ds, sample_count=sample_count, article_key='article', summary_key='highlights')
            datasets.logging.set_verbosity(previous_verbosity)
            return samples
        else:
            datasets.logging.set_verbosity(previous_verbosity)
            raise RuntimeError(f"LỖI: Không thể nạp tập dữ liệu Tiếng Việt (VietNews) từ Hugging Face Hub.")


def generate_oracle_extractive_pairs(articles_data: List[dict], max_pairs: int = 12000) -> List[InputExample]:
    """
    Tự động sinh các cặp câu Oracle (Câu bài báo <-> Câu tóm tắt chuẩn)
    dùng điểm ROUGE-1 làm nhãn (Pull nếu >0.45, Push nếu <0.10)
    """
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    training_examples = []

    print(f"Đang tạo các cặp câu Oracle cho Fine-Tuning (mục tiêu: {max_pairs} cặp).")
    for sample in articles_data:
        if len(training_examples) >= max_pairs:
            break

        article = sample.get('article', '')
        highlights = sample.get('highlights', '')
        if not article or not highlights:
            continue

        try:
            article_sents = nltk.sent_tokenize(article)
            summary_sents = nltk.sent_tokenize(highlights)
        except Exception:
            article_sents = [s.strip() for s in article.split('.') if len(s.strip()) > 10]
            summary_sents = [s.strip() for s in highlights.split('.') if len(s.strip()) > 10]

        for a_sent in article_sents[:10]:  # Tập trung vào các câu mở đầu
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
