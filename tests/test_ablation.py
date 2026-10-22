"""Kiểm tra các hàm ablation trong notebook bằng dữ liệu nhỏ, không gọi Hub."""
import ast
import json
from pathlib import Path
import types

import joblib
import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from src import advanced_ml as ml


def notebook_functions():
    notebook = json.loads((Path(__file__).resolve().parents[1] /
                           'docs/AdvancedML_ExtractiveSummarizer.ipynb').read_text(encoding='utf-8'))
    code = next(''.join(c['source']) for c in notebook['cells']
                if 'def run_feature_ablation(' in ''.join(c['source']))
    tree = ast.parse(code)
    definitions = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)],
                             type_ignores=[])
    namespace = dict(Path=Path, np=np, json=json, joblib=joblib,
                     **{name: getattr(ml, name) for name in (
                         'save_json', 'features', 'labels', 'fit_head', 'select_top',
                         'score_rows', 'write_csv')})
    exec(compile(definitions, '<notebook ablation>', 'exec'), namespace)
    return types.SimpleNamespace(**namespace)


def test_feature_columns_keep_embedding_and_remove_only_requested_extra():
    functions = notebook_functions()
    columns = functions.ablation_columns(4)
    assert columns['embedding_only'] == [0, 1, 2, 3]
    assert columns['embedding_position'] == [0, 1, 2, 3, 4]
    assert columns['embedding_length'] == [0, 1, 2, 3, 5]
    assert columns['embedding_centroid'] == [0, 1, 2, 3, 6]
    assert columns['full'] == list(range(7))
    for name, removed in [('without_position', 4), ('without_length', 5), ('without_centroid', 6)]:
        assert columns[name] == [i for i in range(7) if i != removed]


def test_ablation_reuses_full_head_and_refits_reduced_inputs(tmp_path, monkeypatch):
    functions = notebook_functions()
    rng = np.random.default_rng(42)
    splits, embeddings = {}, {}
    for split, count in [('train', 4), ('validation', 2), ('test', 3)]:
        splits[split], embeddings[split] = [], []
        for i in range(count):
            sentences = ['Học máy phân tích dữ liệu.', 'Mô hình học biểu diễn câu.',
                         'Thời tiết hôm nay có mưa.', 'Đội tuyển thi đấu bóng đá.',
                         'Nghiên cứu thuật toán tối ưu.']
            splits[split].append(dict(id=f'{split}-{i}', sentences=sentences,
                article=' '.join(sentences), reference=' '.join(sentences[:3])))
            emb = rng.normal(size=(5, 4)).astype(np.float32)
            emb /= np.linalg.norm(emb, axis=1, keepdims=True)
            embeddings[split].append(emb)
    corpus = dict(digest='fixture', splits=splits)
    params = {'logistic': dict(family='logistic', C=0.01),
              'mlp': dict(family='mlp', alpha=0.1, solver='adam')}
    selection = {'winners': {f: {'config': p} for f, p in params.items()}}
    scorer = ml.make_rouge_scorer('vi')
    x = np.vstack([ml.features(d, e) for d, e in zip(splits['train'], embeddings['train'])])
    xv = np.vstack([ml.features(d, e) for d, e in zip(splits['validation'], embeddings['validation'])])
    y = np.concatenate(ml.labels(splits['train'], scorer, 3))
    yv = np.concatenate(ml.labels(splits['validation'], scorer, 3))
    full_heads = {}
    with threadpool_limits(limits=1):
        for family, p in params.items():
            full_heads[(family, 42)] = ml.fit_head(x, y, xv, yv, p, 42, max_epochs=1)
        real_labels = functions.run_feature_ablation.__globals__['labels']

        def train_validation_labels_only(docs, scorer, k):
            assert not any(d['id'].startswith('test') for d in docs)
            return real_labels(docs, scorer, k)

        monkeypatch.setitem(functions.run_feature_ablation.__globals__, 'labels', train_validation_labels_only)
        rows = functions.run_feature_ablation(corpus, embeddings, selection, full_heads,
                                              tmp_path, seeds=(42,), max_epochs=1)
        assert len(rows) == 2 * 8 * (2 + 3)
        assert all(r['sentences'] == 3 and len(set(r['selected'].split(','))) == 3 for r in rows)
        assert not list((tmp_path / 'heads').glob('*_full_*'))
        assert len(list((tmp_path / 'heads').glob('*.joblib'))) == 14
        without = joblib.load(tmp_path / 'heads/logistic_without_position_seed42.joblib')
        assert without['scaler'].n_features_in_ == 6
        assert np.allclose(without['scaler'].mean_, x[:, [0, 1, 2, 3, 5, 6]].mean(axis=0))
        assert functions.run_feature_ablation(corpus, embeddings, selection, full_heads,
                    tmp_path, seeds=(42,), max_epochs=1) == rows
        with pytest.raises(ValueError, match='Thiết kế ablation'):
            functions.run_feature_ablation(corpus, embeddings, selection, full_heads,
                                          tmp_path, seeds=(42,), max_epochs=2)


def test_ablation_tables_plot_and_zip_exclude_heavy_cache(tmp_path):
    import matplotlib
    import zipfile
    matplotlib.use('Agg')
    functions = notebook_functions()
    notebook = json.loads((Path(__file__).resolve().parents[1] /
                           'docs/AdvancedML_ExtractiveSummarizer.ipynb').read_text(encoding='utf-8'))
    rows = []
    for split in ('validation', 'test'):
        for family in ('logistic', 'mlp'):
            for variant in functions.ablation_columns(768):
                for seed in (42, 123, 2026):
                    for i in range(4):
                        value = .20 if variant == 'full' else .18
                        rows.append(dict(split=split, family=family, variant=variant,
                            seed=seed, id=f'd{i}', rouge1=value, rouge2=value, rougeL=value,
                            words=50, word_retention=.2))
    ml.write_csv(tmp_path/'original_test_summary.csv', [
        dict(model=f, rougeL_mean=.20) for f in ('logistic', 'mlp')])
    (tmp_path/'embeddings').mkdir()
    (tmp_path/'embeddings/identity.json').write_text('{}')
    (tmp_path/'heads').mkdir()
    (tmp_path/'heads/weights.joblib').write_bytes(b'not-for-export')
    (tmp_path/'corpus.json').write_text('{}')
    namespace = dict(abl_rows=rows, ABL_DIR=tmp_path, ABL_IN_COLAB=False,
        ABL_SEEDS=(42, 123, 2026), ablation_columns=functions.ablation_columns,
        paired_bootstrap=ml.paired_bootstrap, save_json=ml.save_json, Path=Path,
        display=lambda value: None)
    for marker in ('# 13.5.', '# 13.6.'):
        code = next(''.join(c['source']) for c in notebook['cells']
                    if ''.join(c['source']).startswith(marker))
        tree = ast.parse(code)
        # Colab có IPython; test CLI thay riêng thao tác hiển thị, giữ nguyên tính toán/xuất file.
        tree.body = [node for node in tree.body if not (
            isinstance(node, ast.ImportFrom) and node.module == 'IPython.display')]
        exec(compile(tree, '<notebook export>', 'exec'), namespace)
    import pandas as pd
    ci = pd.read_csv(tmp_path/'ablation_bootstrap.csv')
    assert len(ci) == 20
    full_contrasts = ci[ci['left'] == 'full']
    assert np.allclose(full_contrasts.delta_RL_points, 2)
    reproduction = pd.read_csv(tmp_path/'baseline_reproduction.csv')
    assert np.allclose(reproduction.delta_points, 0)
    with zipfile.ZipFile(namespace['result_zip']) as archive:
        names = archive.namelist()
        assert 'figures/feature_ablation.png' in names
        assert 'ablation_bootstrap.csv' in names
        assert not any(n.startswith(('heads/', 'embeddings/')) or n == 'corpus.json' for n in names)
