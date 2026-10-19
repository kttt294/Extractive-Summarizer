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


@pytest.mark.skipif(not RESULTS.exists(), reason='Dữ liệu Colab chỉ có trên máy phát triển')
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


class FakePredictor:
    def summarize(self, text):
        sentences = text.split('\n')
        embeddings = np.eye(len(sentences), 768, dtype=np.float32)
        return sentences, list(range(min(3, len(sentences)))), embeddings


def test_vietnamese_fixed_budget_and_no_silhouette(monkeypatch):
    monkeypatch.setattr(service, 'get_predictor', lambda: FakePredictor())
    text = '\n'.join(f'Câu thứ {i} có đủ thông tin.' for i in range(5))
    for length in ('brief', 'medium', 'detailed'):
        response = SummarizeResponse(**service.process_summarization(text, 'vi', length))
        assert response.stats.summary_sentence_count == 3
        assert response.stats.silhouette_score is None
        assert response.stats.model_version == 'advanced-ml-v2-logistic-seed42'
        assert response.highlight_indices == [0, 1, 2]
        assert len(response.original_sentence_spans) == 5
    short = service.process_summarization('Một câu đủ nội dung.', 'vi')
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
            response = client.post(f'/api/v1/summarize/{route}', json={**payload, 'lang': 'vi'})
            assert response.status_code == 200, response.text
            body = response.json()
            assert body['stats']['summary_sentence_count'] == 3
            assert body['stats']['method'] == 'SBERT + Logistic Regression'
            assert body['stats']['silhouette_score'] is None
        health = client.get('/api/v1/health').json()
        assert health['version'] == '2.0.0'


def test_artifact_failure_never_falls_back_to_old_model(monkeypatch, tmp_path):
    bad = tmp_path/'wrong.joblib'
    bad.write_bytes(b'not-the-deployed-artifact')
    monkeypatch.setenv('VI_HEAD_PATH', str(bad))
    with pytest.raises(RuntimeError, match='không khớp'):
        prediction.AdvancedPredictor()
