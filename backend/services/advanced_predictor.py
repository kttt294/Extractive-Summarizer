"""Serving cấu hình Logistic được chọn trên validation, seed 42."""
import hashlib
import importlib.metadata
import os
import re
from pathlib import Path
from threading import Lock

import joblib
import numpy as np
from src.advanced_ml import encoder_text, features, select_top, split_sentences

BASE_MODEL = 'bkai-foundation-models/vietnamese-bi-encoder'
BASE_REVISION = '84f9d9ada0d1a3c37557398b9ae9fcedcdf40be0'
HEAD_REPO = 'kttt294/vietnamese-extractive-heads-advanced-ml-v2'
HEAD_REVISION = 'cd69d02069d708c2fd8a6b4a78089ff5d1b930b6'
HEAD_SHA256 = '2c183e6a14c3519544695f036ce5c647228ccf475bc36f47fdb7d74c8449a508'


def source_spans(text, sentences):
    """Ánh xạ câu đã xử lý vào văn bản gốc, gồm cả câu bị lọc ở giữa."""
    cursor, spans = 0, []
    for index, sentence in enumerate(sentences):
        start = text.find(sentence, cursor)
        if start >= 0:
            end = start + len(sentence)
        else:
            pattern = r'\s+'.join(re.escape(word) for word in sentence.split())
            match = re.search(pattern, text[cursor:]) if pattern else None
            if match is None:
                continue
            start, end = cursor+match.start(), cursor+match.end()
        # JS String.slice dùng UTF-16, Python dùng code point (khác nhau khi có emoji).
        spans.append(dict(index=index, start=len(text[:start].encode('utf-16-le'))//2,
                          end=len(text[:end].encode('utf-16-le'))//2))
        cursor = end
    return spans


class AdvancedPredictor:
    def __init__(self, encoder=None, fitted=None):
        if fitted is None:
            if importlib.metadata.version('underthesea') != '8.3.0':
                raise RuntimeError('Cần underthesea==8.3.0 để khớp cách tách câu của notebook v2.')
            path = os.getenv('VI_HEAD_PATH')
            if not path:
                from huggingface_hub import hf_hub_download
                path = hf_hub_download(repo_id=HEAD_REPO, filename='heads/logistic_seed42.joblib',
                                       revision=HEAD_REVISION, token=os.getenv('HF_TOKEN') or None)
            path = Path(path)
            if hashlib.sha256(path.read_bytes()).hexdigest() != HEAD_SHA256:
                raise RuntimeError('Checkpoint Logistic không khớp bản v2 seed 42 đã kiểm tra.')
            fitted = joblib.load(path)
        if fitted.get('params') != {'family': 'logistic', 'C': 0.01} or fitted.get('seed') != 42:
            raise RuntimeError('Cấu hình bộ chọn câu khác thí nghiệm đã chọn.')
        if fitted['scaler'].n_features_in_ != 771:
            raise RuntimeError('Scaler cần 768 chiều embedding và ba đặc trưng phụ.')
        self.fitted = fitted
        if encoder is None:
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer(BASE_MODEL, revision=BASE_REVISION,
                                          device=os.getenv('MODEL_DEVICE', 'cpu'))
        self.encoder = encoder
        self.lock = Lock()

    def summarize(self, text):
        sentences = split_sentences(text, 'vi')
        if not sentences:
            raise ValueError('Văn bản không có câu hợp lệ; mỗi câu cần từ 4 đến 150 đơn vị cách trắng.')
        with self.lock:
            emb = np.asarray(self.encoder.encode(
                [encoder_text(s, 'vi') for s in sentences], batch_size=32,
                convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)
            if emb.shape != (len(sentences), 768) or not np.isfinite(emb).all():
                raise RuntimeError('Encoder trả về embedding không hợp lệ.')
            X = self.fitted['scaler'].transform(features({'sentences': sentences}, emb))
            classes = list(self.fitted['model'].classes_)
            scores = self.fitted['model'].predict_proba(X)[:, classes.index(1)]
        selected = select_top(scores, min(3, len(sentences)))
        return sentences, selected, emb


_predictor = None
_load_lock = Lock()


def get_predictor():
    global _predictor
    if _predictor is None:
        with _load_lock:
            if _predictor is None:
                _predictor = AdvancedPredictor()
    return _predictor


def predictor_ready():
    return _predictor is not None
