"""Kiểm tra giao thức bằng dữ liệu tổng hợp; không tải dataset hoặc model."""
import ast
import copy
import csv
import json
import sys
import types
import unicodedata
from pathlib import Path

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from src.advanced_ml import (
    PROTOCOL, add_bertscore, article_hash, features, fit_head, make_plots,
    oracle_indices, paired_bootstrap, predict_head, run_test, run_training,
    save_json, select_top, append_sbert_log, plot_sbert_loss,
)
from src.metrics import VietnameseTokenizer, make_rouge_scorer


def test_vietnamese_rouge_preserves_accents_and_normalizes_unicode():
    scorer = make_rouge_scorer('vi')
    text = 'Trường đại học nghiên cứu ngôn ngữ.'
    assert scorer.score(text, unicodedata.normalize('NFD', text))['rouge1'].fmeasure == 1
    assert VietnameseTokenizer().tokenize('Học_máy, tiếng Việt!') == ['học', 'máy', 'tiếng', 'việt']
    assert scorer.score('má', 'ma')['rouge1'].fmeasure == 0
    assert make_rouge_scorer('en').score('running', 'runs')['rouge1'].fmeasure == 1


def test_oracle_and_tie_break_do_not_repeat_sentences():
    sentences = ['Thể thao có nhiều tin tức.', 'Học máy phân tích dữ liệu.', 'Mô hình dự đoán kết quả.']
    indices = oracle_indices(sentences, 'Học máy phân tích dữ liệu.', make_rouge_scorer('vi'), 1)
    assert indices == [1]
    assert select_top([0.5, 0.5, 0.1], 2) == [0, 1]


def test_document_hash_normalizes_whitespace_case_and_unicode():
    assert article_hash('HỌC  MÁY\n') == article_hash(unicodedata.normalize('NFD', 'học máy'))


def test_features_and_predictions_never_use_reference():
    doc = {'sentences': ['Một câu có bốn từ.', 'Câu khác có năm từ.'], 'reference': 'nhãn bí mật'}
    emb = np.array([[1, 0], [0, 1]], dtype=np.float32)
    before = features(doc, emb)
    doc['reference'] = 'thay đổi reference hoàn toàn'
    np.testing.assert_array_equal(before, features(doc, emb))


def test_scaler_is_fitted_only_on_training_rows():
    X = np.array([[0., 0.], [1., 1.], [2., 0.], [3., 1.]])
    fitted = fit_head(X, np.array([0, 1, 0, 1]), X+1000, np.array([0, 1, 0, 1]),
                      {'family': 'logistic', 'C': 1}, 42)
    np.testing.assert_allclose(fitted['scaler'].mean_, X.mean(axis=0))


def test_bootstrap_direction_and_invalid_pairs():
    result = paired_bootstrap([0.7, 0.8], [0.5, 0.6], repetitions=100)
    assert result['delta'] == pytest.approx(0.2)
    assert result['ci_low'] > 0
    assert paired_bootstrap([1], [1], repetitions=10)['ci_high'] == 0
    with pytest.raises(ValueError):
        paired_bootstrap([1, 2], [1])


def test_sbert_loss_history_keeps_train_and_validation_measurements_separate(tmp_path):
    history = []
    append_sbert_log(tmp_path, history, 0, 0, {'eval_loss': 0.4})
    append_sbert_log(tmp_path, history, 1, 0.01, {'loss': 0.5, 'learning_rate': 0.00001})
    append_sbert_log(tmp_path, history, 50, 0.5, {'loss': 0.2})
    append_sbert_log(tmp_path, history, 50, 0.5, {'eval_loss': 0.25})
    append_sbert_log(tmp_path, history, 100, 1, {'loss': 0.1})
    append_sbert_log(tmp_path, history, 100, 1, {'eval_loss': 0.3})
    append_sbert_log(tmp_path, history, 100, 1, {'train_runtime': 10})
    assert len(history) == 6
    assert history[0]['train_loss'] is None
    assert history[1]['validation_loss'] is None
    assert history[3]['validation_loss'] == 0.25
    stored = json.loads((tmp_path/'finetune_loss_history.json').read_text(encoding='utf-8'))
    assert stored == history
    assert plot_sbert_loss(tmp_path).stat().st_size > 1000


def synthetic_corpus():
    splits, embeddings = {}, {}
    rng = np.random.default_rng(4)
    for split, count in [('train', 8), ('validation', 4), ('test', 4)]:
        docs, arrays = [], []
        for i in range(count):
            sentences = [f'Chủ đề {j} có thông tin quan trọng về nhóm {i}.' for j in range(6)]
            doc = dict(id=f'{split}-{i}', sentences=sentences,
                       reference=' '.join(sentences[:2]), article=' '.join(sentences))
            docs.append(doc)
            emb = rng.normal(size=(6, 5)).astype(np.float32)
            emb /= np.linalg.norm(emb, axis=1, keepdims=True)
            arrays.append(emb)
        splits[split], embeddings[split] = docs, arrays
    return dict(request=dict(lang='vi', k=2), splits=splits, digest='synthetic'), embeddings


def test_offline_end_to_end_selection_lock_exports_and_semantic_cache(tmp_path, monkeypatch):
    corpus, arrays = synthetic_corpus()
    manifest = dict(protocol=PROTOCOL, model='synthetic-only', revision='fixture')
    save_json(tmp_path/'embeddings/pretrained/manifest.json', manifest)
    save_json(tmp_path/'embeddings/finetuned/manifest.json', manifest)
    with threadpool_limits(limits=1):
        selection = run_training(corpus, arrays, tmp_path, seeds=(42, 123), fractions=(0.5, 1.), max_epochs=2)
        assert len(selection['selected']) == 4
        assert run_training(corpus, arrays, tmp_path, seeds=(42, 123), fractions=(0.5, 1.), max_epochs=2) == selection
        rows = run_test(corpus, arrays, selection, tmp_path, finetuned_embeddings=arrays)
    assert len({r['id'] for r in rows}) == 4
    assert all(r['sentences'] == 2 and len(set(r['selected'].split(','))) == 2 for r in rows)
    assert (tmp_path/'paired_bootstrap.csv').exists()
    with (tmp_path/'test_summary.csv').open(encoding='utf-8-sig') as f:
        table = list(csv.DictReader(f))
    assert next(r for r in table if r['model'] == 'mlp')['runs'] == '2'
    changed = copy.deepcopy(selection)
    changed['primary_metric'] = 'rouge1'
    with pytest.raises(ValueError, match='Test đã chạy'):
        run_test(corpus, arrays, changed, tmp_path, finetuned_embeddings=arrays)
    calls = []

    class FakeBERTScorer:
        hash = 'offline-test-only'

        def __init__(self, **kwargs):
            pass

        def score(self, hypotheses, references, **kwargs):
            calls.append(len(hypotheses))
            return None, None, types.SimpleNamespace(cpu=lambda: np.full(len(hypotheses), 0.8))

    monkeypatch.setitem(sys.modules, 'bert_score', types.SimpleNamespace(BERTScorer=FakeBERTScorer))
    add_bertscore(corpus, tmp_path, batch_size=4)
    count = len(calls)
    add_bertscore(corpus, tmp_path, batch_size=4)
    assert len(calls) == count
    with (tmp_path/'test_summary.csv').open(encoding='utf-8-sig') as f:
        assert all(float(r['bertscore_f1_mean']) == pytest.approx(0.8) for r in csv.DictReader(f))
    make_plots(tmp_path)
    assert len(list((tmp_path/'figures').glob('*.png'))) == 4


def test_training_does_not_read_test_documents(tmp_path):
    corpus, arrays = synthetic_corpus()

    class NoTestAccess:
        def __iter__(self):
            raise AssertionError('Không được truy cập test trong huấn luyện')

    corpus['splits']['test'] = NoTestAccess()
    arrays['test'] = NoTestAccess()
    save_json(tmp_path/'embeddings/pretrained/manifest.json', {'model': 'fixture'})
    with threadpool_limits(limits=1):
        run_training(corpus, arrays, tmp_path, seeds=(42,), fractions=(1.,), max_epochs=1)


def test_notebook_is_one_sequential_workflow_with_github_setup():
    root = Path(__file__).resolve().parents[1]
    new = json.loads((root/'AdvancedML_ExtractiveSummarizer.ipynb').read_text(encoding='utf-8'))
    source = '\n'.join(''.join(c['source']) for c in new['cells'])
    assert 'https://github.com/kttt294/Extractive-Summarizer.git' in source
    for obsolete in ('RUN_LEGACY', 'smoke', 'files.upload', 'drive.mount', '/content/drive', 'Phụ lục'):
        assert obsolete not in source
    for cell in new['cells']:
        if cell['cell_type'] == 'code':
            tree = ast.parse(''.join(cell['source']))
            assert not any(isinstance(node, ast.Name) and node.id == 'MODE' for node in ast.walk(tree))
    stages = [source.index(f'{name}(') for name in (
        'prepare_corpus', 'finetune_encoder', 'cache_embeddings',
        'run_training', 'run_test', 'add_bertscore', 'make_plots')]
    assert stages == sorted(stages)
    final_cell = ''.join(new['cells'][-1]['source'])
    assert 'shutil.make_archive(' in final_cell or 'zipfile.ZipFile(' in final_cell
    assert 'files.download(result_zip)' in final_cell
