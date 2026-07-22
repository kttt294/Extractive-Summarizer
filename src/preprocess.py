import nltk
from langdetect import detect
from src.config import OPTIMAL_HYPERPARAMS

# Tải bộ tách câu của NLTK (punkt & punkt_tab cho NLTK 3.9+)
for resource in ['punkt', 'punkt_tab']:
    try:
        nltk.data.find(f'tokenizers/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)

try:
    from underthesea import sent_tokenize as sent_tokenize_vi
except ImportError:
    # Dự phòng nếu chưa cài underthesea
    def sent_tokenize_vi(text):
        return [s.strip() for s in text.split('.') if s.strip()]


def resolve_language(text: str, user_choice: str = 'auto') -> str:
    """
    Tự động phát hiện ngôn ngữ ('en' hoặc 'vi') bằng langdetect
    hoặc trả về lựa chọn thủ công của người dùng.
    """
    if user_choice in ['en', 'vi']:
        return user_choice
    
    try:
        detected = detect(text)
        return 'en' if detected == 'en' else 'vi'
    except Exception:
        return 'vi'  # Mặc định tiếng Việt nếu có lỗi


def preprocess_text(text: str, lang: str = 'vi'):
    """
    Tách câu và lọc câu dựa trên độ dài từ.
    Trả về: Danh sách các tuple [(chỉ_số_gốc, nội_dung_câu)]
    """
    if lang == 'en':
        try:
            raw_sentences = nltk.sent_tokenize(text)
        except Exception:
            raw_sentences = [s.strip() for s in text.split('.') if s.strip()]
    else:
        try:
            raw_sentences = sent_tokenize_vi(text)
        except Exception:
            raw_sentences = [s.strip() for s in text.split('.') if s.strip()]

    min_w = OPTIMAL_HYPERPARAMS['min_words']
    max_w = OPTIMAL_HYPERPARAMS['max_words']

    filtered_sentences = []
    for idx, sent in enumerate(raw_sentences):
        cleaned = sent.strip()
        word_count = len(cleaned.split())
        
        # Loại bỏ URL, các câu quá ngắn hoặc quá dài
        if min_w <= word_count <= max_w and not cleaned.startswith(('http://', 'https://')):
            filtered_sentences.append((idx, cleaned))

    return filtered_sentences
