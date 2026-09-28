import os
from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from src.config import MODEL_CONFIGS

_LOADED_MODELS = {}

def get_sbert_model(lang: str = 'vi', use_finetuned: bool = False) -> SentenceTransformer:
    """
    Lazy loading mô hình SentenceTransformer (Pretrained hoặc Fine-Tuned)
    """
    key = f"{lang}_{'finetuned' if use_finetuned else 'pretrained'}"
    if key in _LOADED_MODELS:
        return _LOADED_MODELS[key]

    if use_finetuned:
        model_path = MODEL_CONFIGS['finetuned_vi'] if lang == 'vi' else MODEL_CONFIGS['finetuned_en']
        if (model_path.startswith('./') or model_path.startswith('.\\') or os.path.isabs(model_path)) and not os.path.exists(model_path):
            print(f"Không tìm thấy mô hình Fine-tuned tại {model_path}. Tự động chuyển sang mô hình Pretrained.")
            model_path = MODEL_CONFIGS[lang]
    else:
        model_path = MODEL_CONFIGS[lang]

    try:
        model = SentenceTransformer(model_path)
    except Exception as e:
        print(f"Lỗi tải mô hình {model_path}: {e}. Đang thử tải mô hình Pretrained mặc định...")
        model_path = MODEL_CONFIGS[lang]
        model = SentenceTransformer(model_path)

    _LOADED_MODELS[key] = model
    return model


HF_API_MODEL = os.getenv("HF_API_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
HF_TOKEN = os.getenv("HF_TOKEN", "")


def embed_sentences(sentences: List[Tuple[int, str]], lang: str = 'vi', use_finetuned: bool = False) -> np.ndarray:
    """
    Trích xuất vector nhúng SBERT cho danh sách các tuple (vị_trí_câu, câu_văn_bản)
    Ưu tiên gọi Serverless Inference API (Hugging Face) để tiết kiệm 100% RAM/GPU trên Cloud.
    Tự động fallback về mô hình PyTorch cục bộ nếu không có mạng/API lỗi.
    Trả về: Numpy array kích thước (N, 768) đã chuẩn hóa L2 (chuẩn = 1).
    """
    if not sentences:
        return np.array([])
    texts = [text for _, text in sentences]
    if lang == 'vi':
        try:
            from underthesea import word_tokenize
            texts = [word_tokenize(t, format="text") for t in texts]
        except ImportError:
            pass

    # 1. Thử gọi qua Hugging Face Serverless API (Nhẹ, không tốn RAM, phù hợp Cloud Free)
    if HF_TOKEN:
        try:
            from huggingface_hub import InferenceClient
            client = InferenceClient(token=HF_TOKEN)
            res = client.feature_extraction(texts, model=HF_API_MODEL)
            embeddings = np.array(res)
            # Chuẩn hóa L2 Norm để đảm bảo chuẩn = 1 cho Cosine Similarity & K-Means
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-12)
            return embeddings
        except Exception as e:
            print(f"HF Inference API lưu ý: {e}. Đang chuyển sang chạy mô hình nội bộ...")

    # 2. Fallback: Tải và chạy mô hình cục bộ bằng PyTorch/SentenceTransformers
    model = get_sbert_model(lang=lang, use_finetuned=use_finetuned)
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
    return embeddings

