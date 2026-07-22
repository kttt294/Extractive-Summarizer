import os
from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from src.config import MODEL_CONFIGS

_LOADED_MODELS = {}

def get_sbert_model(lang: str = 'vi', use_finetuned: bool = False) -> SentenceTransformer:
    """
    Nạp mô hình SentenceTransformer theo dạng Singleton (Pretrained hoặc Fine-Tuned).
    """
    key = f"{lang}_{'finetuned' if use_finetuned else 'pretrained'}"
    if key in _LOADED_MODELS:
        return _LOADED_MODELS[key]

    if use_finetuned:
        model_path = MODEL_CONFIGS['finetuned_vi'] if lang == 'vi' else MODEL_CONFIGS['finetuned_en']
        if not os.path.exists(model_path):
            print(f"Không tìm thấy mô hình Fine-tuned tại {model_path}. Tự động chuyển sang mô hình Pretrained gốc.")
            model_path = MODEL_CONFIGS[lang]
    else:
        model_path = MODEL_CONFIGS[lang]

    print(f"Đang nạp mô hình SBERT: {model_path} ...")
    model = SentenceTransformer(model_path)
    _LOADED_MODELS[key] = model
    return model


def embed_sentences(sentences: List[Tuple[int, str]], lang: str = 'vi', use_finetuned: bool = False) -> np.ndarray:
    """
    Trích xuất vector nhúng SBERT cho danh sách các tuple (chỉ_số_gốc, văn_bản).
    Trả về: Numpy array kích thước (N, 768) hoặc (N, 384).
    """
    if not sentences:
        return np.array([])
    
    texts = [text for _, text in sentences]
    model = get_sbert_model(lang=lang, use_finetuned=use_finetuned)
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings
