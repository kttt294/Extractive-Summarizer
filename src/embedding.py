import os
from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from src.config import MODEL_CONFIGS

_LOADED_MODELS = {}

def get_sbert_model(lang: str = 'vi', use_finetuned: bool = False) -> SentenceTransformer:
    """
    Singleton loader for SentenceTransformer models (Pretrained or Fine-Tuned).
    """
    key = f"{lang}_{'finetuned' if use_finetuned else 'pretrained'}"
    if key in _LOADED_MODELS:
        return _LOADED_MODELS[key]

    if use_finetuned:
        model_path = MODEL_CONFIGS['finetuned_vi'] if lang == 'vi' else MODEL_CONFIGS['finetuned_en']
        if not os.path.exists(model_path):
            print(f"Fine-tuned model not found at {model_path}. Fallback to pretrained model.")
            model_path = MODEL_CONFIGS[lang]
    else:
        model_path = MODEL_CONFIGS[lang]

    print(f"Loading SBERT model: {model_path} ...")
    model = SentenceTransformer(model_path)
    _LOADED_MODELS[key] = model
    return model


def embed_sentences(sentences: List[Tuple[int, str]], lang: str = 'vi', use_finetuned: bool = False) -> np.ndarray:
    """
    Generates SBERT sentence embeddings for a list of (original_index, text) tuples.
    Returns: numpy array of shape (N, 768) or (N, 384)
    """
    if not sentences:
        return np.array([])
    
    texts = [text for _, text in sentences]
    model = get_sbert_model(lang=lang, use_finetuned=use_finetuned)
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings
