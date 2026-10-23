"""Kiểm tra serving, hợp đồng API và highlight; không gọi Hub trong test."""
import hashlib
import csv
import json
from pathlib import Path

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.services import advanced_predictor as prediction
from backend.services import summarizer_service as service
from backend.schemas.payload import SummarizeResponse

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT/'advanced_ml_v2_vi_results'


def test_source_spans_track_repeated_sentences_and_emoji():
    text = '📰 Tin ngắn.\nMột câu đầy đủ thông tin.\nMột câu đầy đủ thông tin.'
    sentences = ['Một câu đầy đủ thông tin.', 'Một câu đầy đủ thông tin.']
    spans = prediction.source_spans(text, sentences)
    raw = text.encode('utf-16-le')
    assert len(spans) == 2
    for item in spans:
        assert raw[item['start']*2:item['end']*2].decode('utf-16-le') == sentences[item['index']]
    assert spans[0]['end'] < spans[1]['start']


@pytest.mark.skipif(
    not all((RESULTS / name).is_file() for name in (
        'heads/f7aa9d91fb34ea95976c.joblib',
        'corpus.json', 'test_per_document.csv', 'embeddings/pretrained/test.npz',
    )),
    reason='Cần weights và cache embedding của lượt Colab để đối chiếu artifact thật',
)
def test_real_artifact_predictions_match_saved_notebook_results(monkeypatch):
    head = RESULTS/'heads/f7aa9d91fb34ea95976c.joblib'
    assert hashlib.sha256(head.read_bytes()).hexdigest() == prediction.HEAD_SHA256
    fitted = joblib.load(head)
    corpus = json.loads((RESULTS/'corpus.json').read_text(encoding='utf-8'))
    # Thay encoder bằng embedding thật đã lưu; vẫn dùng nguyên head và scaler thật.
    docs = corpus['splits']['test'][:12]
    with (RESULTS/'test_per_document.csv').open(encoding='utf-8-sig') as file:
        reference = {r['id']: [int(i) for i in r['selected'].split(',')]
                     for r in csv.DictReader(file) if r['model'] == 'logistic' and r['seed'] == '42'}
    arrays = np.load(RESULTS/'embeddings/pretrained/test.npz')

    class Encoder:
        def encode(self, texts, **kwargs):
            assert kwargs['normalize_embeddings']
            return self.emb

    encoder = Encoder()
    predictor = prediction.AdvancedPredictor(encoder=encoder, fitted=fitted)
    for i, doc in enumerate(docs):
        encoder.emb = arrays[f'd{i}']
        sentences, indices, _ = predictor.summarize(doc['article'])
        assert sentences == doc['sentences']
        assert indices == reference[doc['id']]
    arrays.close()


def test_predictor_length_changes_budget_and_preserves_source_order(monkeypatch):
    sentences = [f'Câu thứ {i} có đầy đủ thông tin.' for i in range(4)]
    monkeypatch.setattr(prediction, 'split_sentences', lambda text, lang: sentences)
    monkeypatch.setattr(prediction, 'encoder_text', lambda text, lang: text)

    class Encoder:
        def encode(self, texts, **kwargs):
            return np.eye(len(texts), 768, dtype=np.float32)

    class Scaler:
        def transform(self, values):
            return values

    class Model:
        classes_ = np.array([0, 1])

        def predict_proba(self, values):
            scores = np.array([0.2, 0.8, 0.5, 0.9])
            return np.column_stack([1 - scores, scores])

    predictor = prediction.AdvancedPredictor.__new__(prediction.AdvancedPredictor)
    predictor.encoder = Encoder()
    predictor.fitted = {'scaler': Scaler(), 'model': Model()}
    predictor.lock = prediction.Lock()
    for count, expected in [(2, [1, 3]), (3, [1, 2, 3]), (5, [0, 1, 2, 3])]:
        _, selected, _ = predictor.summarize(' '.join(sentences), k=count)
        assert selected == expected


class FakePredictor:
    def summarize(self, text, k=3):
        sentences = text.split('\n')
        embeddings = np.eye(len(sentences), 768, dtype=np.float32)
        return sentences, list(range(min(k, len(sentences)))), embeddings


def test_vietnamese_length_modes_and_no_silhouette(monkeypatch):
    monkeypatch.setattr(service, 'get_predictor', lambda: FakePredictor())
    text = '\n'.join(f'Câu thứ {i} có đủ thông tin.' for i in range(5))
    for length, count in [('brief', 2), ('medium', 3), ('detailed', 5)]:
        response = SummarizeResponse(**service.process_summarization(text, 'vi', length))
        assert response.stats.summary_sentence_count == count
        assert response.stats.summary_policy == f'top-{count}'
        assert response.stats.silhouette_score is None
        assert response.stats.model_version == 'advanced-ml-v2-logistic-seed42'
        assert response.highlight_indices == list(range(count))
        assert len(response.original_sentence_spans) == 5
    for length in ('brief', 'medium', 'detailed'):
        short = service.process_summarization('Một câu đủ nội dung.', 'vi', length)
        assert short['stats']['summary_sentence_count'] == 1
        assert short['stats']['diversity_score'] is None


def test_api_and_all_input_routes_use_same_v2_service(monkeypatch):
    from backend.main import app
    from backend.routers import summarize
    monkeypatch.setenv('PRELOAD_MODELS', 'false')
    monkeypatch.setattr(service, 'get_predictor', lambda: FakePredictor())
    text = '\n'.join(f'Câu thứ {i} có thông tin đầy đủ.' for i in range(4))
    monkeypatch.setattr(summarize, 'crawl_article_from_url', lambda url: {'text': text, 'title': 'Bài báo'})
    monkeypatch.setattr(summarize, 'crawl_articles_from_topic', lambda *args, **kwargs:
                        {'combined_text': text, 'is_today_news': True, 'date_range': 'Hôm nay', 'notice': None})
    with TestClient(app) as client:
        cases = [('text', {'text': text}), ('url', {'url': 'https://example.com/article'}), ('topic', {'topic': 'học máy'})]
        for route, payload in cases:
            for length, count in [('brief', 2), ('medium', 3), ('detailed', 4)]:
                response = client.post(f'/api/v1/summarize/{route}', json={**payload, 'lang': 'vi', 'length': length})
                assert response.status_code == 200, response.text
                body = response.json()
                assert body['stats']['summary_sentence_count'] == count
                assert body['stats']['method'] == 'SBERT + Logistic Regression'
                assert body['stats']['silhouette_score'] is None
        health = client.get('/api/v1/health').json()
        assert health['version'] == '2.0.0'


def test_artifact_failure_never_falls_back_to_old_model(monkeypatch, tmp_path):
    # Kiểm tra riêng checksum, không phụ thuộc phiên bản tokenizer của máy chạy test.
    original_version = prediction.importlib.metadata.version
    monkeypatch.setattr(prediction.importlib.metadata, 'version',
                        lambda name: '8.3.0' if name == 'underthesea' else original_version(name))
    bad = tmp_path/'wrong.joblib'
    bad.write_bytes(b'not-the-deployed-artifact')
    monkeypatch.setenv('VI_HEAD_PATH', str(bad))
    with pytest.raises(RuntimeError, match='không khớp'):
        prediction.AdvancedPredictor()
