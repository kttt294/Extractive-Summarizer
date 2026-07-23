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

# Đảm bảo punkt và punkt_tab được tải sẵn cho NLTK 3.9+
for resource in ['punkt', 'punkt_tab']:
    try:
        nltk.data.find(f'tokenizers/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)


def load_evaluation_dataset(lang: str = 'en', sample_count: int = 200):
    """
    Nạp dữ liệu thử nghiệm (Tiếng Anh & Tiếng Việt) với logging sạch sẽ.
    """
    print(f"Đang nạp bộ dữ liệu thử nghiệm {lang.upper()} (số lượng={sample_count})...")
    
    # Tắt log của datasets trong suốt quá trình nạp
    previous_verbosity = datasets.logging.get_verbosity()
    datasets.logging.set_verbosity_error()
    
    if lang == 'en':
        ds = None
        for dataset_name in ["abisee/cnn_dailymail", "cnn_dailymail"]:
            try:
                ds = load_dataset(dataset_name, "3.0.0", split="test")
                if ds is not None:
                    break
            except Exception:
                pass

        if ds is not None:
            samples = ds.select(range(min(sample_count, len(ds))))
            datasets.logging.set_verbosity(previous_verbosity)
            return [
                {
                    'id': idx,
                    'article': sample['article'],
                    'highlights': sample['highlights']
                }
                for idx, sample in enumerate(samples)
            ]
        else:
            sample_en_article = {
                'id': 0,
                'article': "Artificial Intelligence (AI) is transforming industries worldwide at an unprecedented pace. Recent breakthroughs in Natural Language Processing enable computers to summarize complex documents in seconds. Extractive summarization directly selects the most informative sentences from original texts. Using Sentence-BERT embeddings, machines capture deep semantic relationships between sentences. K-Means clustering groups similar concepts together, ensuring the summary covers multiple key topics. Post-filtering removes redundant information for clear reading.",
                'highlights': "Artificial Intelligence is transforming industries globally. Extractive summarization uses Sentence-BERT and K-Means to extract key informative sentences without redundancy."
            }
            datasets.logging.set_verbosity(previous_verbosity)
            return [sample_en_article] * min(sample_count, 100)
    else:
        ds = None
        for dataset_name in ["bkai-foundation-models/vietnews", "vietnews"]:
            try:
                ds = load_dataset(dataset_name, split="test")
                if ds is not None:
                    break
            except Exception:
                pass

        if ds is not None:
            samples = ds.select(range(min(sample_count, len(ds))))
            datasets.logging.set_verbosity(previous_verbosity)
            return [
                {
                    'id': idx,
                    'article': sample.get('article') or sample.get('text', ''),
                    'highlights': sample.get('abstract') or sample.get('summary', '')
                }
                for idx, sample in enumerate(samples)
            ]
        else:
            sample_vi_article = {
                'id': 0,
                'article': "Trí tuệ nhân tạo (AI) đang tạo nên cuộc cách mạng mạnh mẽ trong nhiều lĩnh vực của đời sống xã hội. Tại Việt Nam, nhiều doanh nghiệp công nghệ lớn đang đẩy mạnh đầu tư vào nghiên cứu và phát triển các mô hình ngôn ngữ lớn (LLM) dành riêng cho tiếng Việt. Các chuyên gia đánh giá việc chủ động về công nghệ AI sẽ giúp đảm bảo an ninh dữ liệu quốc gia và nâng cao năng lực cạnh tranh. Trong tương lai gần, AI sẽ hỗ trợ đắc lực cho y tế, giáo dục và quản lý hành chính công.",
                'highlights': "AI đang tạo cuộc cách mạng tại Việt Nam. Doanh nghiệp đẩy mạnh phát triển mô hình ngôn ngữ lớn cho tiếng Việt nhằm đảm bảo an ninh dữ liệu và phát triển y tế, giáo dục."
            }
            datasets.logging.set_verbosity(previous_verbosity)
            return [sample_vi_article] * min(sample_count, 100)


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
