import os

# Đường dẫn bộ các hình Pretrained và Fine-Tuned trên Hugging Face & Local
MODEL_CONFIGS = {
    'en': 'sentence-transformers/all-mpnet-base-v2',
    'vi': 'bkai-foundation-models/vietnamese-bi-encoder',
    'finetuned_vi': 'kttt294/vietnamese-sbert-finetuned',
    'finetuned_en': 'kttt294/english-sbert-finetuned'
}

# Các siêu tham số tối ưu (Thu được từ quá trình Grid Search Evaluation)
OPTIMAL_HYPERPARAMS_VI = {
    'alpha': 0.10,        # Tỷ lệ chọn câu K% (Việt)
    'theta': 0.70,        # Ngưỡng Cosine Similarity để lọc trùng ngữ nghĩa (Việt)
    'lambda': 0.20,       # Trọng số ưu tiên vị trí câu đầu bài (Việt)
    'min_words': 4,       
    'max_words': 90,      
    'buffer_k': 2         
}

OPTIMAL_HYPERPARAMS_EN = {
    'alpha': 0.05,        # Tỷ lệ chọn câu K% (Anh)
    'theta': 0.65,        # Ngưỡng Cosine Similarity để lọc trùng ngữ nghĩa (Anh)
    'lambda': 0.50,       # Trọng số ưu tiên vị trí câu đầu bài (Anh)
    'min_words': 4,       
    'max_words': 90,      
    'buffer_k': 2         
}

# Khởi tạo mặc định là Tiếng Việt
OPTIMAL_HYPERPARAMS = OPTIMAL_HYPERPARAMS_VI.copy()

def set_language_config(lang: str):
    """
    Tự động chuyển đổi bộ siêu tham số cho phù hợp với đặc thù phân phối của ngôn ngữ
    """
    global OPTIMAL_HYPERPARAMS
    if lang == 'en':
        OPTIMAL_HYPERPARAMS.update(OPTIMAL_HYPERPARAMS_EN)
    else:
        OPTIMAL_HYPERPARAMS.update(OPTIMAL_HYPERPARAMS_VI)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
