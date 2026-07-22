import os

# Đường dẫn bộ mô hìnhPretrained và Fine-Tuned trên Hugging Face & Local
MODEL_CONFIGS = {
    'en': 'sentence-transformers/all-MiniLM-L6-v2',
    'vi': 'bkai-foundation-models/vietnamese-bi-encoder',
    'finetuned_vi': './models/finetuned_sbert_vi',
    'finetuned_en': './models/finetuned_sbert_en'
}

# Các siêu tham số tối ưu (Thu được từ quá trình Grid Search Evaluation)
OPTIMAL_HYPERPARAMS = {
    'alpha': 0.25,        # Tỷ lệ chọn câu K%
    'theta': 0.85,        # Ngưỡng Cosine Similarity để lọc trùng ngữ nghĩa (Post-filtering)
    'min_words': 8,       # Số từ tối thiểu cho một câu hợp lệ
    'max_words': 60,      # Số từ tối đa cho một câu hợp lệ
    'buffer_k': 2         # Số lượng cụm đệm cho K-Means để bù đắp sau khi lọc trùng
}

# Thư mục lưu trữ dự án
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
