"""Thực nghiệm có train/validation/test độc lập cho Học máy nâng cao.

Không dùng API embedding, không tự đổi mô hình khi tải lỗi, không chọn cấu hình
bằng test. Notebook điều phối từng giai đoạn; file kết quả dùng cho báo cáo.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import time
import unicodedata
from pathlib import Path

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from src.metrics import make_rouge_scorer

PROTOCOL = 'aml-v1-unicode-rouge-fixed-k'


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def article_hash(text):
    return hashlib.sha256(' '.join(unicodedata.normalize('NFC', text).lower().split()).encode()).hexdigest()


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def split_sentences(text, lang):
    if lang == 'vi':
        from underthesea import sent_tokenize
    else:
        from nltk import sent_tokenize
    # Một quy tắc chung cho mọi mô hình; không thay theo ngôn ngữ bằng global config.
    sentences = []
    for line in text.splitlines():
        for sent in sent_tokenize(line.strip()) if line.strip() else []:
            sent = sent.strip()
            if 4 <= len(sent.split()) <= 150 and not sent.startswith(('http://', 'https://')):
                sentences.append(sent)
    return sentences


def prepare_corpus(root, dataset_name, lang='vi', counts=None, subset=None, seed=42, revision=None, k=3):
    """Lấy mẫu ngẫu nhiên từ các split chính thức; lưu snapshot để chạy lại.

    Kiểm tra trùng chính xác sau chuẩn hóa ở cả trong và giữa các split.
    Không tuyên bố loại được các bản tin gần trùng/paraphrase.
    """
    from datasets import load_dataset
    from huggingface_hub import HfApi

    root = Path(root)
    counts = counts or {'train': 3000, 'validation': 500, 'test': 1000}
    request = dict(protocol=PROTOCOL, dataset=dataset_name, subset=subset, lang=lang,
                   counts=counts, seed=seed, revision=revision, k=k)
    path = root / 'corpus.json'
    if path.exists():
        cached = json.loads(path.read_text(encoding='utf-8'))
        if cached['request'] != request:
            raise ValueError('Cấu hình dữ liệu đã đổi. Hãy dùng thư mục run mới.')
        return cached
    if set(counts) != {'train', 'validation', 'test'} or min(counts.values()) < 1 or k < 1:
        raise ValueError('Cần ba split với số lượng dương và k >= 1.')
    revision = HfApi().dataset_info(dataset_name, revision=revision).sha
    seen = set()
    splits, audit = {}, {}
    for split in ('train', 'validation', 'test'):
        ds = load_dataset(dataset_name, subset, split=split, revision=revision)
        fingerprint = ds._fingerprint
        ds = ds.shuffle(seed=seed)
        docs, dropped = [], 0
        for row in ds:
            article = row.get('article') or row.get('text') or row.get('document')
            reference = row.get('highlights') or row.get('summary') or row.get('abstract')
            if not isinstance(article, str) or not isinstance(reference, str) or not reference.strip():
                continue
            doc_id = article_hash(article)
            if doc_id in seen:
                dropped += 1
                continue
            sentences = split_sentences(article, lang)
            if len(sentences) <= k:
                continue
            seen.add(doc_id)
            docs.append(dict(id=doc_id, article=article, reference=reference, sentences=sentences))
            if len(docs) == counts[split]:
                break
        if len(docs) != counts[split]:
            raise ValueError(f'{split}: chỉ đủ {len(docs)}/{counts[split]} mẫu hợp lệ.')
        splits[split] = docs
        audit[split] = dict(count=len(docs), duplicates_skipped=dropped, fingerprint=fingerprint)
        print(f'{split}: {len(docs)} bài; bỏ {dropped} bản trùng chính xác.', flush=True)
    corpus = dict(request=request, resolved_revision=revision, audit=audit, splits=splits)
    corpus['digest'] = digest(splits)
    save_json(path, corpus)
    return corpus


def oracle_indices(sentences, reference, scorer, k):
    """Greedy oracle xấp xỉ: tối đa hóa ROUGE-1 + ROUGE-2, đủ k câu.

    Chỉ dùng để tạo nhãn train/validation, không phải nhãn người gán.
    """
    selected = []
    for _ in range(min(k, len(sentences))):
        candidates = []
        for i in range(len(sentences)):
            if i in selected:
                continue
            summary = ' '.join(sentences[j] for j in sorted(selected + [i]))
            scores = scorer.score(reference, summary)
            candidates.append((scores['rouge1'].fmeasure + scores['rouge2'].fmeasure, -i, i))
        selected.append(max(candidates)[2])
    return sorted(selected)


def finetune_encoder(corpus, root, base_model, epochs=3, batch_size=32,
                     learning_rate=2e-5, weight_decay=0.01, seed=42):
    """Fine-tune mới, chỉ từ train; evaluator chỉ đọc validation.

    Nhãn cặp là ROUGE-1 liên tục giữa câu nguồn và câu tham chiếu.
    Mỗi bài lấy tối đa 8 cặp trải đều theo điểm để tránh vài bài chiếm hết dữ liệu.
    """
    # Trainer có thể tự bật W&B nếu Colab cài sẵn; chỉ lưu log cục bộ.
    os.environ['WANDB_MODE'] = 'disabled'
    import torch
    from huggingface_hub import HfApi
    from sentence_transformers import (InputExample, SentenceTransformer, losses,
                                       SentenceTransformerTrainer, SentenceTransformerTrainingArguments)
    from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
    from datasets import Dataset
    from transformers import TrainerCallback

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    root = Path(root)
    path = root / 'encoder_finetuned'
    settings = dict(corpus=corpus['digest'], base_model=base_model, epochs=epochs,
                    batch_size=batch_size, learning_rate=learning_rate,
                    weight_decay=weight_decay, seed=seed, pair_policy='rouge1-quantiles-8-per-doc',
                    training_protocol='trainer-loss-curves-v2')
    manifest = root / 'finetune_manifest.json'
    if manifest.exists():
        saved = json.loads(manifest.read_text(encoding='utf-8'))
        if saved['settings'] != settings or not (path / 'modules.json').exists():
            raise ValueError('Checkpoint không khớp cấu hình. Hãy dùng run mới.')
        return str(path.resolve())
    lang = corpus['request']['lang']
    scorer = make_rouge_scorer(lang, ('rouge1',))

    def pairs(split):
        examples = []
        rng = np.random.default_rng(seed)
        for doc in corpus['splits'][split]:
            refs = split_sentences(doc['reference'], lang) or [doc['reference']]
            candidates = []
            for sent in doc['sentences']:
                scores = [scorer.score(ref, sent)['rouge1'].fmeasure for ref in refs]
                best = int(np.argmax(scores))
                candidates.append((float(scores[best]), sent, refs[best]))
            candidates.sort(key=lambda x: x[0])
            indices = np.unique(np.linspace(0, len(candidates)-1, min(8, len(candidates))).astype(int))
            for idx in indices:
                label, a, b = candidates[idx]
                examples.append(InputExample(texts=[encoder_text(a, lang), encoder_text(b, lang)], label=label))
        rng.shuffle(examples)
        return examples

    train, validation = pairs('train'), pairs('validation')
    revision = HfApi().model_info(base_model).sha
    model = SentenceTransformer(base_model, revision=revision)
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(validation, name='validation', write_csv=True)
    def as_dataset(examples):
        return Dataset.from_dict(dict(sentence1=[x.texts[0] for x in examples],
                                      sentence2=[x.texts[1] for x in examples],
                                      score=[float(x.label) for x in examples]))

    history = []

    class LossCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            append_sbert_log(root, history, state.global_step, state.epoch, logs or {})

    steps_per_epoch = max(1, (len(train) + batch_size - 1) // batch_size)
    eval_steps = max(1, steps_per_epoch // 4)
    args = SentenceTransformerTrainingArguments(
        output_dir=str(root/'finetune_checkpoints'), num_train_epochs=epochs,
        per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate, weight_decay=weight_decay, warmup_ratio=0.1,
        optim='adamw_torch', seed=seed, data_seed=seed,
        logging_strategy='steps', logging_steps=min(50, eval_steps), logging_first_step=True,
        eval_strategy='steps', eval_steps=eval_steps,
        save_strategy='steps', save_steps=eval_steps, save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model='eval_loss', greater_is_better=False,
        report_to='none', fp16=False, bf16=False)
    trainer = SentenceTransformerTrainer(
        model=model, args=args, train_dataset=as_dataset(train), eval_dataset=as_dataset(validation),
        loss=losses.CosineSimilarityLoss(model), evaluator=evaluator, callbacks=[LossCallback()])
    started = time.perf_counter()
    trainer.evaluate()  # Mốc validation trước khi cập nhật trọng số, tại step 0.
    trainer.train()
    # Đây là checkpoint đã được Trainer khôi phục theo validation MSE thấp nhất.
    model.save(str(path))
    save_json(root/'finetune_trainer_history.json', trainer.state.log_history)
    plot_sbert_loss(root)
    save_json(manifest, dict(settings=settings, base_revision=revision,
                            train_pairs=len(train), validation_pairs=len(validation),
                            seconds=time.perf_counter()-started,
                            selection='minimum validation cosine MSE; not test or summarization ROUGE',
                            best_validation_loss=float(trainer.state.best_metric),
                            best_checkpoint=trainer.state.best_model_checkpoint,
                            eval_steps=eval_steps, logging_steps=min(50, eval_steps)))
    del trainer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return str(path.resolve())


def append_sbert_log(root, history, step, epoch, logs):
    """Lưu điểm đo thật; không nội suy loss giữa các lần đánh giá."""
    if not any(key in logs for key in ('loss', 'eval_loss')):
        return
    def number(value):
        return float(value) if value is not None and np.isfinite(float(value)) else None
    history.append(dict(step=int(step), epoch=number(epoch),
                        train_loss=number(logs.get('loss')),
                        validation_loss=number(logs.get('eval_loss')),
                        learning_rate=number(logs.get('learning_rate')),
                        grad_norm=number(logs.get('grad_norm'))))
    write_csv(Path(root)/'finetune_loss_history.csv', history)
    save_json(Path(root)/'finetune_loss_history.json', history)


def plot_sbert_loss(root):
    """Train: loss trung bình cửa sổ minibatch; validation: MSE toàn bộ cặp held-out."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(root)
    history = json.loads((root/'finetune_loss_history.json').read_text(encoding='utf-8'))
    train = [row for row in history if row['train_loss'] is not None]
    validation = [row for row in history if row['validation_loss'] is not None]
    if not train or not validation:
        raise ValueError('Cần có cả training loss và validation loss để vẽ SBERT.')
    folder = root/'figures'
    folder.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot([r['step'] for r in train], [r['train_loss'] for r in train],
            label='Training MSE (minibatch window mean)', alpha=0.8)
    ax.plot([r['step'] for r in validation], [r['validation_loss'] for r in validation],
            marker='o', label='Validation MSE (held-out pairs)')
    best = min((r for r in validation if r['step'] > 0), key=lambda r: r['validation_loss'], default=None)
    if best:
        ax.axvline(best['step'], color='gray', linestyle='--',
                   label=f"Lowest measured validation MSE: step {best['step']}")
    ax.set(xlabel='Optimizer step', ylabel='Cosine similarity MSE loss', title='SBERT fine-tuning loss')
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(folder/'sbert_loss_curve.png', dpi=200)
    plt.close(fig)
    return folder/'sbert_loss_curve.png'


def encoder_text(text, lang):
    if lang == 'vi':
        from underthesea import word_tokenize
        return word_tokenize(text, format='text')
    return text


def cache_embeddings(corpus, root, model_name, tag='pretrained'):
    """Đóng băng encoder, ghi model/revision và hash dữ liệu vào cache."""
    import torch
    from huggingface_hub import HfApi
    from sentence_transformers import SentenceTransformer

    root = Path(root)
    folder = root / 'embeddings' / tag
    folder.mkdir(parents=True, exist_ok=True)
    local = Path(model_name).is_dir()
    # Hash nội dung checkpoint cục bộ để không tái dùng cache sau khi thay weights.
    if local:
        hasher = hashlib.sha256()
        for p in sorted(Path(model_name).rglob('*')):
            if p.is_file() and p.suffix in ('.json', '.safetensors', '.bin'):
                hasher.update(str(p.relative_to(model_name)).encode())
                with p.open('rb') as f:
                    for block in iter(lambda: f.read(1024*1024), b''):
                        hasher.update(block)
        revision = hasher.hexdigest()
    else:
        revision = HfApi().model_info(model_name).sha
    identity = dict(protocol=PROTOCOL, corpus=corpus['digest'], model=str(model_name),
                    revision=revision, normalized=True, vietnamese_word_tokenization=True)
    manifest = folder / 'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8')) != identity:
        raise ValueError(f'Cache {tag} không còn khớp. Dùng thư mục run mới.')
    save_json(manifest, identity)
    model, result = None, {}
    for split, docs in corpus['splits'].items():
        cache = folder / f'{split}.npz'
        if cache.exists():
            with np.load(cache, allow_pickle=False) as data:
                arrays = [data[f'd{i}'] for i in range(len(docs))]
        else:
            if model is None:
                model = SentenceTransformer(model_name, **({} if local else {'revision': revision}))
                save_json(folder/'encoder_stats.json', dict(
                    total_parameters=sum(p.numel() for p in model.parameters()),
                    embedding_dimension=model.get_sentence_embedding_dimension(),
                    max_sequence_length=model.max_seq_length,
                    device=str(model.device),
                    note='Head experiments freeze all encoder parameters.'))
            texts = [encoder_text(s, corpus['request']['lang']) for d in docs for s in d['sentences']]
            started = time.perf_counter()
            encoded = model.encode(texts, batch_size=32, convert_to_numpy=True,
                                   normalize_embeddings=True, show_progress_bar=True)
            arrays, offset = [], 0
            for doc in docs:
                size = len(doc['sentences'])
                arrays.append(encoded[offset:offset+size].astype(np.float32))
                offset += size
            np.savez_compressed(cache, **{f'd{i}': x for i, x in enumerate(arrays)})
            save_json(folder / f'{split}_timing.json', dict(seconds=time.perf_counter()-started,
                                                          sentences=len(texts)))
        if any(len(x) != len(d['sentences']) or not np.isfinite(x).all() for x, d in zip(arrays, docs)):
            raise ValueError(f'Cache không hợp lệ: {cache}')
        result[split] = arrays
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def features(doc, emb):
    """Cùng đặc trưng cho Logistic/MLP; không đọc reference."""
    n = len(emb)
    centroid = emb.mean(axis=0)
    centroid /= max(np.linalg.norm(centroid), 1e-12)
    extras = np.column_stack((np.arange(n) / max(n-1, 1),
                              np.log1p([len(s.split()) for s in doc['sentences']]), emb @ centroid))
    return np.column_stack((emb, extras)).astype(np.float32)


def labels(docs, scorer, k):
    output = []
    for doc in docs:
        y = np.zeros(len(doc['sentences']), dtype=np.int64)
        y[oracle_indices(doc['sentences'], doc['reference'], scorer, k)] = 1
        output.append(y)
    return output


def select_top(scores, k):
    # Stable tie-break theo thứ tự câu, sau đó khôi phục thứ tự văn bản.
    return sorted(np.argsort(-np.asarray(scores), kind='stable')[:k].tolist())


def summary_from(doc, indices):
    return ' '.join(doc['sentences'][i] for i in sorted(indices))


def score_rows(docs, predictions, scorer, model, seed):
    rows = []
    for doc, selected in zip(docs, predictions):
        text = summary_from(doc, selected)
        scores = scorer.score(doc['reference'], text)
        rows.append(dict(id=doc['id'], model=model, seed=seed,
                         **{name: float(score.fmeasure) for name, score in scores.items()},
                         sentences=len(selected), words=len(text.split()),
                         word_retention=len(text.split()) / max(1, len(doc['article'].split())),
                         selected=','.join(map(str, selected))))
    return rows


def fit_head(X, y, Xv, yv, params, seed, max_epochs=40):
    """Scaler chỉ fit train. Early stopping dùng validation theo bài báo đã tách."""
    scaler = StandardScaler().fit(X)
    X, Xv = scaler.transform(X), scaler.transform(Xv)
    started = time.perf_counter()
    trace = []
    if params['family'] == 'logistic':
        model = LogisticRegression(C=params['C'], solver='lbfgs', max_iter=1000, random_state=seed)
        model.fit(X, y)
        trace.append(dict(epoch=1, train_logloss=float(log_loss(y, model.predict_proba(X), labels=[0, 1])),
                          validation_logloss=float(log_loss(yv, model.predict_proba(Xv), labels=[0, 1]))))
    else:
        model = MLPClassifier(hidden_layer_sizes=(128,), activation='relu', solver=params['solver'],
                              alpha=params['alpha'], learning_rate_init=0.001,
                              batch_size=min(256, len(X)), shuffle=True,
                              random_state=seed, early_stopping=False)
        best, best_loss, stale = None, float('inf'), 0
        for epoch in range(1, max_epochs+1):
            model.partial_fit(X, y, classes=np.array([0, 1]))
            loss = float(log_loss(yv, model.predict_proba(Xv), labels=[0, 1]))
            trace.append(dict(epoch=epoch,
                              train_logloss=float(log_loss(y, model.predict_proba(X), labels=[0, 1])),
                              validation_logloss=loss))
            if loss < best_loss - 1e-5:
                best, best_loss, stale = copy.deepcopy(model), loss, 0
            else:
                stale += 1
            if stale >= 5:
                break
        model = best
    train_seconds = time.perf_counter()-started
    parameter_count = int(sum(x.size for x in model.coefs_) + sum(x.size for x in model.intercepts_)) if params['family'] == 'mlp' else int(model.coef_.size+model.intercept_.size)
    return dict(model=model, scaler=scaler, trace=trace, train_seconds=train_seconds,
                parameters=parameter_count, params=params, seed=seed)


def predict_head(fitted, docs, embeddings, k):
    return [select_top(fitted['model'].predict_proba(fitted['scaler'].transform(features(d, e)))[:, 1], k)
            for d, e in zip(docs, embeddings)]


def run_training(corpus, embeddings, root, seeds=(42, 123, 2026), fractions=(0.25, 0.5, 1.0), max_epochs=40):
    """Tìm cấu hình bằng seed đầu; chạy lặp và learning curve sau khi chọn.

    Không truy cập test. Lưu tất cả thử nghiệm và validation ROUGE.
    """
    root = Path(root)
    settings = dict(protocol=PROTOCOL, corpus=corpus['digest'], seeds=list(seeds),
                    fractions=list(fractions), max_epochs=max_epochs,
                    encoder=json.loads((root/'embeddings/pretrained/manifest.json').read_text(encoding='utf-8')))
    selection_path = root / 'selection.json'
    if selection_path.exists():
        saved = json.loads(selection_path.read_text(encoding='utf-8'))
        if saved['settings'] != settings:
            raise ValueError('Thiết lập train đã thay đổi; hãy dùng run mới.')
        return saved
    scorer = make_rouge_scorer(corpus['request']['lang'])
    k = corpus['request']['k']
    train, val = corpus['splits']['train'], corpus['splits']['validation']
    y, yv = labels(train, scorer, k), labels(val, scorer, k)
    save_json(root/'oracle_label_audit.json', {
        name: dict(documents=len(targets), sentences=sum(len(t) for t in targets),
                   positive_sentences=int(sum(t.sum() for t in targets)),
                   positive_fraction=float(np.concatenate(targets).mean()))
        for name, targets in [('train', y), ('validation', yv)]})
    X = [features(d, e) for d, e in zip(train, embeddings['train'])]
    Xv, yv = np.vstack([features(d, e) for d, e in zip(val, embeddings['validation'])]), np.concatenate(yv)
    experiments, fitted_cache = [], {}

    def run(params, seed, fraction):
        key = digest(dict(params=params, seed=seed, fraction=fraction, settings=settings))[:20]
        count = max(1, int(len(train)*fraction))
        artifact = root / 'heads' / f'{key}.joblib'
        artifact.parent.mkdir(parents=True, exist_ok=True)
        if artifact.exists():
            fitted = joblib.load(artifact)
        else:
            fitted = fit_head(np.vstack(X[:count]), np.concatenate(y[:count]), Xv, yv, params, seed, max_epochs)
            joblib.dump(fitted, artifact)
        pred = predict_head(fitted, val, embeddings['validation'], k)
        rows = score_rows(val, pred, scorer, key, seed)
        rouge = float(np.mean([r['rougeL'] for r in rows]))
        item = dict(key=key, family=params['family'], seed=seed, fraction=fraction, train_documents=count,
                    validation_rougeL=rouge, parameters=fitted['parameters'],
                    train_seconds=fitted['train_seconds'], params=json.dumps(params, sort_keys=True))
        experiments.append(item)
        write_csv(root/'traces'/f'{key}.csv', fitted['trace'])
        print(f"{params} seed={seed} train={count}: validation ROUGE-L={rouge:.4f}", flush=True)
        fitted_cache[(json.dumps(params, sort_keys=True), seed, fraction)] = key
        return item

    candidates = [dict(family='logistic', C=c) for c in (0.01, 1.0, 100.0)]
    candidates += [dict(family='mlp', alpha=a, solver=s) for s in ('adam', 'sgd') for a in (0.0, 0.001, 0.1)]
    winners = {}
    for params in candidates:
        row = run(params, seeds[0], 1.0)
        family = params['family']
        if family not in winners or row['validation_rougeL'] > winners[family]['validation_rougeL']:
            winners[family] = dict(row, config=params)
    selected = []
    for family, winner in winners.items():
        params = winner['config']
        for seed in seeds:
            for fraction in fractions:
                cache_key = (json.dumps(params, sort_keys=True), seed, fraction)
                key = fitted_cache.get(cache_key)
                if key is None:
                    key = run(params, seed, fraction)['key']
                if fraction == 1.0:
                    selected.append(dict(family=family, seed=seed, key=key))
    write_csv(root/'validation_experiments.csv', experiments)
    selection = dict(settings=settings, winners=winners, selected=selected,
                     primary_metric='rougeL', selection_seed=seeds[0],
                     caveat='Hyperparameters selected using first seed; remaining seeds measure replication, not nested CV.')
    save_json(selection_path, selection)
    return selection


def baseline_predictions(docs, arrays, kind, k, seed=42):
    predictions = []
    for doc, emb in zip(docs, arrays):
        if kind == 'lead':
            chosen = list(range(min(k, len(emb))))
        elif kind == 'centroid':
            chosen = select_top(emb @ emb.mean(axis=0), k)
        elif kind == 'textrank':
            import networkx as nx
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            matrix = TfidfVectorizer().fit_transform(doc['sentences'])
            sims = cosine_similarity(matrix)
            np.fill_diagonal(sims, 0.0)
            scores = nx.pagerank(nx.from_numpy_array(sims), max_iter=1000)
            chosen = select_top([scores[i] for i in range(len(emb))], k)
        elif kind == 'kmeans':
            fitted = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(emb)
            chosen = []
            for i, center in enumerate(fitted.cluster_centers_):
                members = np.flatnonzero(fitted.labels_ == i)
                if len(members):
                    chosen.append(int(members[np.argmin(np.linalg.norm(emb[members]-center, axis=1))]))
            # Trường hợp embedding trùng làm số cụm thực tế < k.
            for idx in select_top(emb @ emb.mean(axis=0), len(emb)):
                if len(chosen) >= k:
                    break
                if idx not in chosen:
                    chosen.append(idx)
        else:
            raise ValueError(kind)
        predictions.append(sorted(chosen))
    return predictions


def paired_bootstrap(a, b, seed=42, repetitions=2000):
    """CI 95% cho trung bình (a-b); lấy mẫu theo bài báo, không theo câu."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError('Hai dãy điểm phải có cùng kích thước.')
    differences = a-b
    if differences.ndim != 1 or len(differences) == 0 or not np.isfinite(differences).all():
        raise ValueError('Cần hai dãy điểm hữu hạn, cùng kích thước và không rỗng.')
    rng = np.random.default_rng(seed)
    means = [float(np.mean(rng.choice(differences, len(differences), replace=True))) for _ in range(repetitions)]
    return dict(delta=float(differences.mean()), ci_low=float(np.quantile(means, 0.025)),
                ci_high=float(np.quantile(means, 0.975)))


def run_test(corpus, embeddings, selection, root, finetuned_embeddings=None):
    """Chỉ chạy sau khi khóa selection. Test không huấn luyện hay chọn cấu hình."""
    root = Path(root)
    lock = dict(selection=digest(selection), corpus=corpus['digest'], protocol=PROTOCOL,
                finetuned=(json.loads((root/'embeddings/finetuned/manifest.json').read_text(encoding='utf-8'))
                           if finetuned_embeddings is not None else None))
    lock_path = root/'test_lock.json'
    if lock_path.exists() and json.loads(lock_path.read_text(encoding='utf-8')) != lock:
        raise ValueError('Test đã chạy với cấu hình khác. Không lựa chọn lại bằng test; dùng run mới có ghi lý do.')
    save_json(lock_path, lock)
    docs = corpus['splits']['test']
    k = corpus['request']['k']
    scorer = make_rouge_scorer(corpus['request']['lang'])
    rows, timing = [], []
    seeds = selection['settings']['seeds']

    def evaluate(name, seed, call):
        started = time.perf_counter()
        predicted = call()
        seconds = time.perf_counter()-started
        if len(predicted) != len(docs) or any(len(p) != k or len(set(p)) != k for p in predicted):
            raise ValueError(f'{name}: không đủ ngân sách {k} câu.')
        rows.extend(score_rows(docs, predicted, scorer, name, seed))
        timing.append(dict(model=name, seed=seed, documents=len(docs),
                           selector_seconds=seconds, selector_ms_per_document=seconds/len(docs)*1000))
        write_csv(root/'test_per_document.csv', rows)
        print(f'Đã đánh giá {name}, seed={seed}', flush=True)

    for kind in ('lead', 'textrank', 'centroid', 'kmeans'):
        for seed in seeds if kind == 'kmeans' else [seeds[0]]:
            evaluate('pretrained_'+kind, seed, lambda kind=kind, seed=seed:
                     baseline_predictions(docs, embeddings['test'], kind, k, seed))
    if finetuned_embeddings is not None:
        for kind in ('centroid', 'kmeans'):
            for seed in seeds if kind == 'kmeans' else [seeds[0]]:
                evaluate('finetuned_'+kind, seed, lambda kind=kind, seed=seed:
                         baseline_predictions(docs, finetuned_embeddings['test'], kind, k, seed))
    for entry in selection['selected']:
        fitted = joblib.load(root/'heads'/f"{entry['key']}.joblib")
        evaluate(entry['family'], entry['seed'], lambda fitted=fitted:
                 predict_head(fitted, docs, embeddings['test'], k))
    write_csv(root/'test_selector_timing.csv', timing)
    aggregate_results(rows, root)
    return rows


def aggregate_results(rows, root):
    root = Path(root)
    metrics = ['rouge1', 'rouge2', 'rougeL', 'words', 'word_retention']
    if rows and 'bertscore_f1' in rows[0]:
        metrics.append('bertscore_f1')
    models = sorted(set(r['model'] for r in rows))
    tables, per_doc = [], {}
    for model in models:
        subset = [r for r in rows if r['model'] == model]
        seeds = sorted(set(r['seed'] for r in subset))
        item = dict(model=model, runs=len(seeds))
        for metric in metrics:
            means = [np.mean([r[metric] for r in subset if r['seed'] == seed]) for seed in seeds]
            item[metric+'_mean'] = float(np.mean(means))
            item[metric+'_seed_std'] = float(np.std(means, ddof=1)) if len(means)>1 else None
        tables.append(item)
        per_doc[model] = {doc_id: float(np.mean([r['rougeL'] for r in subset if r['id'] == doc_id]))
                          for doc_id in sorted(set(r['id'] for r in subset))}
    write_csv(root/'test_summary.csv', tables)
    comparisons = []
    for model in models:
        if model == 'pretrained_kmeans':
            continue
        ids = sorted(per_doc[model])
        if ids != sorted(per_doc['pretrained_kmeans']):
            raise ValueError('Không có cùng tập bài báo để paired bootstrap.')
        comparisons.append(dict(model=model, baseline='pretrained_kmeans', metric='rougeL',
                                **paired_bootstrap([per_doc[model][i] for i in ids],
                                                   [per_doc['pretrained_kmeans'][i] for i in ids])))
    write_csv(root/'paired_bootstrap.csv', comparisons)
    save_json(root/'completion.json', dict(protocol=PROTOCOL, documents=len(per_doc[models[0]]),
                                         models=models, uncertainty='Seed std across fitted heads; bootstrap across documents after seed averaging. CIs are unadjusted, exploratory.'))
    for item in tables:
        print(f"{item['model']}: ROUGE-L={item['rougeL_mean']:.4f}, seed std={item['rougeL_seed_std']}")


def add_bertscore(corpus, root, batch_size=16):
    """BERTScore độc lập với encoder đang so sánh; lỗi được báo thay vì ghi 0.

    Tính một lần cho mỗi bản tóm tắt khác nhau để tiết kiệm thời gian/GPU.
    """
    from bert_score import BERTScorer
    root = Path(root)
    with (root/'test_per_document.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    docs = {d['id']: d for d in corpus['splits']['test']}
    lang = corpus['request']['lang']
    model_name = 'bert-base-multilingual-cased' if lang == 'vi' else 'roberta-large'
    scorer = BERTScorer(model_type=model_name, lang=lang, rescale_with_baseline=False)
    keys = list(dict.fromkeys((r['id'], r['selected']) for r in rows))
    cache_path = root/'bertscore_cache.json'
    signature = dict(model=model_name, lang=lang, corpus=corpus['digest'], rescale_with_baseline=False)
    cached = json.loads(cache_path.read_text(encoding='utf-8')) if cache_path.exists() else dict(signature=signature, scores={})
    if cached['signature'] != signature:
        raise ValueError('Cache BERTScore không khớp.')
    pending = [key for key in keys if digest(key) not in cached['scores']]
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start+batch_size]
        hypotheses = [summary_from(docs[doc_id], [int(i) for i in selected.split(',')]) for doc_id, selected in batch]
        references = [docs[doc_id]['reference'] for doc_id, _ in batch]
        _, _, values = scorer.score(hypotheses, references, batch_size=batch_size)
        for key, value in zip(batch, values.cpu().tolist()):
            cached['scores'][digest(key)] = float(value)
        save_json(cache_path, cached)
        if start % (batch_size*25) == 0:
            print(f'BERTScore: {start+len(batch)}/{len(pending)} bản tóm tắt còn thiếu', flush=True)
    for row in rows:
        row['seed'] = int(row['seed'])
        for name in ('rouge1', 'rouge2', 'rougeL', 'words', 'word_retention', 'sentences'):
            row[name] = float(row[name])
        row['bertscore_f1'] = cached['scores'][digest((row['id'], row['selected']))]
    write_csv(root/'test_per_document.csv', rows)
    save_json(root/'bertscore_metadata.json', dict(**signature, hashcode=scorer.hash,
                                                note='Contextual token matching; long inputs may be truncated by the scorer.'))
    aggregate_results(rows, root)


def make_plots(root):
    """Đồ thị trực tiếp từ CSV thực nghiệm, không nhập số liệu thủ công."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    root = Path(root)
    folder = root/'figures'
    folder.mkdir(exist_ok=True)
    with (root/'validation_experiments.csv').open(encoding='utf-8-sig') as f:
        experiments = list(csv.DictReader(f))
    selection = json.loads((root/'selection.json').read_text(encoding='utf-8'))
    fig, ax = plt.subplots(figsize=(7, 4))
    for family, winner in selection['winners'].items():
        params = json.dumps(winner['config'], sort_keys=True)
        subset = [r for r in experiments if r['params'] == params]
        fractions = sorted(set(float(r['fraction']) for r in subset))
        means, deviations = [], []
        for fraction in fractions:
            scores = [float(r['validation_rougeL']) for r in subset if float(r['fraction']) == fraction]
            means.append(np.mean(scores))
            deviations.append(np.std(scores, ddof=1) if len(scores)>1 else 0)
        ax.errorbar(fractions, means, yerr=deviations, marker='o', capsize=4, label=family)
    ax.set(xlabel='Fraction of training documents', ylabel='Validation ROUGE-L F1')
    ax.legend()
    fig.tight_layout()
    fig.savefig(folder/'learning_curve.png', dpi=200)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for entry in selection['selected']:
        if entry['seed'] != selection['settings']['seeds'][0] or entry['family'] != 'mlp':
            continue
        with (root/'traces'/f"{entry['key']}.csv").open(encoding='utf-8-sig') as f:
            trace = list(csv.DictReader(f))
        for metric in ('train_logloss', 'validation_logloss'):
            ax.plot([int(r['epoch']) for r in trace], [float(r[metric]) for r in trace], label=metric)
    ax.set(xlabel='Epoch', ylabel='Binary cross-entropy')
    ax.legend()
    fig.tight_layout()
    fig.savefig(folder/'mlp_convergence.png', dpi=200)
    plt.close(fig)
    with (root/'test_summary.csv').open(encoding='utf-8-sig') as f:
        summary = list(csv.DictReader(f))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh([r['model'] for r in summary], [float(r['rougeL_mean']) for r in summary])
    ax.set(xlabel='Test ROUGE-L F1 (mean over seeds)')
    fig.tight_layout()
    fig.savefig(folder/'test_rougeL.png', dpi=200)
    plt.close(fig)
    # Regularization và optimizer, seed tìm cấu hình, dữ liệu train đầy đủ.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    search = [r for r in experiments if int(r['seed']) == selection['selection_seed'] and float(r['fraction']) == 1.0]
    logistic = [r for r in search if r['family'] == 'logistic']
    axes[0].plot([json.loads(r['params'])['C'] for r in logistic],
                 [float(r['validation_rougeL']) for r in logistic], marker='o')
    axes[0].set(xscale='log', xlabel='Logistic C (inverse L2 strength)', ylabel='Validation ROUGE-L F1')
    for solver in ('adam', 'sgd'):
        subset = [r for r in search if r['family'] == 'mlp' and json.loads(r['params'])['solver'] == solver]
        axes[1].plot([json.loads(r['params'])['alpha'] for r in subset],
                     [float(r['validation_rougeL']) for r in subset], marker='o', label=solver)
    axes[1].set(xscale='symlog', xlabel='MLP L2 alpha', ylabel='Validation ROUGE-L F1')
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(folder/'regularization_optimizer.png', dpi=200)
    plt.close(fig)


def record_environment(root):
    import torch
    packages = ('torch', 'numpy', 'scikit-learn', 'sentence-transformers', 'transformers',
                'datasets', 'rouge-score', 'underthesea', 'networkx')
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = 'missing'
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
               for p in Path(__file__).parent.glob('*.py')}
    path = Path(root)/'environment.json'
    if path.exists():
        previous = json.loads(path.read_text(encoding='utf-8'))
        if previous['source_hashes'] != sources or previous['packages'] != versions:
            raise ValueError('Mã nguồn hoặc phiên bản thư viện đã đổi. Đặt RUN_NAME mới để tránh trộn kết quả cache.')
        return
    save_json(path, dict(python=platform.python_version(), packages=versions,
                                               cuda=torch.cuda.is_available(),
                                               gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                                               source_hashes=sources))
