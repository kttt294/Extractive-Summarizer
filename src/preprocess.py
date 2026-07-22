import nltk
from langdetect import detect
from src.config import OPTIMAL_HYPERPARAMS

# Download NLTK tokenizer models if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    from underthesea import sent_tokenize as sent_tokenize_vi
except ImportError:
    # Fallback to splitlines if underthesea is not installed
    def sent_tokenize_vi(text):
        return [s.strip() for s in text.split('.') if s.strip()]


def resolve_language(text: str, user_choice: str = 'auto') -> str:
    """
    Detects language ('en' or 'vi') automatically using langdetect
    or returns user_choice if specified.
    """
    if user_choice in ['en', 'vi']:
        return user_choice
    
    try:
        detected = detect(text)
        return 'en' if detected == 'en' else 'vi'
    except Exception:
        return 'vi'  # Default to Vietnamese on failure


def preprocess_text(text: str, lang: str = 'vi'):
    """
    Sentence tokenization and filtering based on sentence length.
    Returns: List of tuples [(original_index, cleaned_sentence_text)]
    """
    if lang == 'en':
        raw_sentences = nltk.sent_tokenize(text)
    else:
        raw_sentences = sent_tokenize_vi(text)

    min_w = OPTIMAL_HYPERPARAMS['min_words']
    max_w = OPTIMAL_HYPERPARAMS['max_words']

    filtered_sentences = []
    for idx, sent in enumerate(raw_sentences):
        cleaned = sent.strip()
        word_count = len(cleaned.split())
        
        # Filter out URLs, very short or excessively long sentences
        if min_w <= word_count <= max_w and not cleaned.startswith(('http://', 'https://')):
            filtered_sentences.append((idx, cleaned))

    return filtered_sentences
