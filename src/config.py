import os

# Đường dẫn bộ các hình Pretrained và Fine-Tuned trên Hugging Face & Local
MODEL_CONFIGS = {
    'en': 'sentence-transformers/all-MiniLM-L6-v2',
    'vi': 'bkai-foundation-models/vietnamese-bi-encoder',
    'finetuned_vi': 'kttt294/vietnamese-sbert-finetuned',
    'finetuned_en': './models/finetuned_sbert_en'
}

# Các siêu tham số tối ưu (Thu được từ quá trình Grid Search Evaluation)
OPTIMAL_HYPERPARAMS = {
    'alpha': 0.25,        # Tỷ lệ chọn câu K%
    'theta': 0.88,        # Ngưỡng Cosine Similarity để lọc trùng ngữ nghĩa (Post-filtering)
    'min_words': 4,       # Số từ tối thiểu cho một câu hợp lệ (Giữ được tiêu đề & lọc cụm rác <4 từ)
    'max_words': 90,      # Số từ tối đa cho một câu hợp lệ (Bao quát được các câu báo chí phức hợp)
    'buffer_k': 2         # Số lượng cụm đệm cho K-Means để bù đắp sau khi lọc trùng
}


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
