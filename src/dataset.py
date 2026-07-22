import nltk
from typing import List, Tuple
from datasets import load_dataset
from rouge_score import rouge_scorer
from sentence_transformers import InputExample

# Đảm bảo punkt được tải sẵn
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


def load_evaluation_dataset(lang: str = 'en', sample_count: int = 200):
    """
    Nạp dữ liệu thử nghiệm với Fallback URI mới nhất của Hugging Face Hub:
    - Tiếng Anh: abisee/cnn_dailymail hoặc cnn_dailymail v3.0.0
    - Tiếng Việt: bkai-foundation-models/vietnews hoặc vietnews
    """
    print(f"Đang nạp bộ dữ liệu thử nghiệm {lang.upper()} (số lượng={sample_count})...")
    if lang == 'en':
        ds = None
        # Thử nạp theo tên repository chuẩn mới của HF Hub
        try:
            ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test")
        except Exception:
            try:
                ds = load_dataset("cnn_dailymail", "3.0.0", split="test")
            except Exception as err:
                print(f"Lỗi nạp dataset cnn_dailymail: {err}")
                return []

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
        ds = None
        try:
            ds = load_dataset("bkai-foundation-models/vietnews", split="test")
        except Exception:
            try:
                ds = load_dataset("vietnews", split="test")
            except Exception:
                pass

        if ds is not None:
            samples = ds.select(range(min(sample_count, len(ds))))
            return [
                {
                    'id': idx,
                    'article': sample.get('article') or sample.get('text', ''),
                    'highlights': sample.get('abstract') or sample.get('summary', '')
                }
                for idx, sample in enumerate(samples)
            ]
        else:
            print("Chưa nạp trực tiếp được dataset VietNews. Đang tạo dữ liệu mẫu Tiếng Việt...")
            # Dữ liệu mẫu fallback Tiếng Việt để huấn luyện không bị gián đoạn
            sample_vietnamese_articles = [
                {
                    'id': 0,
                    'article': "Trí tuệ nhân tạo (AI) đang tạo nên cuộc cách mạng mạnh mẽ trong nhiều lĩnh vực của đời sống xã hội. Tại Việt Nam, nhiều doanh nghiệp công nghệ lớn đang đẩy mạnh đầu tư vào nghiên cứu và phát triển các mô hình ngôn ngữ lớn (LLM) dành riêng cho tiếng Việt. Các chuyên gia đánh giá việc chủ động về công nghệ AI sẽ giúp đảm bảo an ninh dữ liệu quốc gia và nâng cao năng lực cạnh tranh. Trong tương lai gần, AI sẽ hỗ trợ đắc lực cho y tế, giáo dục và quản lý hành chính công.",
                    'highlights': "AI đang tạo cuộc cách mạng tại Việt Nam. Doanh nghiệp đẩy mạnh phát triển mô hình ngôn ngữ lớn cho tiếng Việt nhằm đảm bảo an ninh dữ liệu và phát triển y tế, giáo dục."
                }
            ] * min(sample_count, 100)
            return sample_vietnamese_articles


def generate_oracle_extractive_pairs(articles_data: List[dict], max_pairs: int = 12000) -> List[InputExample]:
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

        article = sample.get('article', '')
        highlights = sample.get('highlights', '')
        if not article or not highlights:
            continue

        article_sents = nltk.sent_tokenize(article)
        summary_sents = nltk.sent_tokenize(highlights)

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
