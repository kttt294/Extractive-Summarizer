"""Tạo notebook end-to-end, giữ nguyên notebook NLP gốc để đối chiếu."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def md(text):
    return dict(cell_type='markdown', metadata={}, source=text.splitlines(keepends=True))


def code(text, **metadata):
    return dict(cell_type='code', metadata=metadata, source=text.splitlines(keepends=True),
                execution_count=None, outputs=[])


cells = [
md('''# Tóm tắt văn bản trích xuất cho môn Học máy nâng cao

Notebook end-to-end: chuẩn bị dữ liệu → fine-tune SBERT → trích xuất embedding →
Logistic Regression/MLP → lựa chọn mô hình → đánh giá test → xuất bảng và hình.

**Câu hỏi nghiên cứu:** regularization, optimizer và lượng dữ liệu ảnh hưởng thế nào đến
bộ chọn câu trên embedding đóng băng? Fine-tuning encoder có cải thiện chất lượng
khi giữ nguyên thuật toán chọn câu và cùng ngân sách ba câu không?

Các bước của bài NLP được kế thừa tại vị trí tương ứng, cập nhật cho thực nghiệm mới.
Không đưa các điểm ROUGE tiếng Việt cũ vào bảng mới:
thực nghiệm mới giữ nguyên Unicode có dấu và dùng quy trình lấy mẫu mới.

**Colab:** chọn Runtime → Change runtime type → T4 GPU. Chạy lần lượt từ trên xuống.
Cell đầu tự clone hoặc cập nhật repository GitHub. Chọn Run all để chạy toàn bộ.
'''),
code('''# 1. Lấy mã nguồn từ GitHub, giống notebook NLP trước đây.
import os
import sys
import subprocess
from pathlib import Path

REPO_URL = 'https://github.com/kttt294/Extractive-Summarizer.git'
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    repo_dir = Path('/content/Extractive-Summarizer')
    if not (repo_dir / '.git').exists():
        subprocess.run(['git', 'clone', REPO_URL, str(repo_dir)], check=True)
    else:
        subprocess.run(['git', '-C', str(repo_dir), 'pull', '--ff-only'], check=True)
    os.chdir(repo_dir)
assert Path('src/advanced_ml.py').exists(), 'Hãy chạy notebook từ thư mục repository.'
sys.path.insert(0, str(Path.cwd()))
print('Thư mục mã nguồn:', Path.cwd())
'''),
code('''# 2. Cài môi trường. Chỉ cài thư viện thực nghiệm, không cần frontend/backend.
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements-ml.txt'], check=True)
import nltk
for resource in ('punkt', 'punkt_tab'):
    if not nltk.download(resource, quiet=True):
        raise RuntimeError(f'Không tải được tokenizer {resource}')
import torch
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
'''),
md('''## 3. Cấu hình thực nghiệm

Tiếng Việt là thực nghiệm chính: 3.000 bài train, 500 validation và 1.000 test.
Notebook chạy lần lượt toàn bộ các bước với một cấu hình duy nhất.
Kết quả và checkpoint được lưu trên Google Drive để giữ lại khi phiên Colab ngắt.
Có thể đổi `LANG = 'en'` để chạy thêm tiếng Anh.
'''),
code('''LANG = 'vi'
RUN_NAME = 'advanced_ml_v1'
if LANG not in ('vi', 'en'):
    raise ValueError('LANG phải là vi hoặc en.')
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    OUTPUT_BASE = Path('/content/drive/MyDrive/NLP_BTL_HocMayNangCao')
else:
    OUTPUT_BASE = Path('outputs/advanced_ml').resolve()
RUN_DIR = OUTPUT_BASE / f'{RUN_NAME}_{LANG}'
COUNTS = {'train': 3000, 'validation': 500, 'test': 1000}
SEEDS = (42, 123, 2026)
FRACTIONS = (0.25, 0.5, 1.0)
MAX_EPOCHS = 40
FT_EPOCHS = 3
BASE_MODEL = ('bkai-foundation-models/vietnamese-bi-encoder' if LANG == 'vi'
              else 'sentence-transformers/all-mpnet-base-v2')
DATASET = 'nam194/vietnews' if LANG == 'vi' else 'abisee/cnn_dailymail'
SUBSET = None if LANG == 'vi' else '3.0.0'
RUN_DIR.mkdir(parents=True, exist_ok=True)
print('Kết quả sẽ lưu tại:', RUN_DIR)
print('Quy mô:', COUNTS)
'''),
code('''# 4. Nạp các hàm và lưu cấu hình thực nghiệm.
from src.advanced_ml import (
    prepare_corpus, finetune_encoder, cache_embeddings, run_training,
    run_test, add_bertscore, make_plots, record_environment, save_json,
)
record_environment(RUN_DIR)
save_json(RUN_DIR / 'run_config.json', dict(lang=LANG, counts=COUNTS,
          seeds=list(SEEDS), fractions=list(FRACTIONS), max_epochs=MAX_EPOCHS,
          ft_epochs=FT_EPOCHS, base_model=BASE_MODEL, dataset=DATASET, k=3))
'''),
md('''## 5. Cố định dữ liệu

Dùng train/validation/test chính thức, shuffle seed 42, lấy đủ số bài hợp lệ.
Mỗi bài phải có hơn ba câu sau cùng một quy tắc lọc độ dài (4–150 đơn vị cách trắng).
Loại bài trùng chính xác sau chuẩn hóa trong và giữa các tập; chưa loại gần trùng.
Lưu bài, bản tóm tắt tham chiếu, câu đã tách, ID băm, revision và fingerprint.
Không chia ngẫu nhiên các câu của cùng một bài sang các tập khác nhau.
'''),
code('''corpus = prepare_corpus(RUN_DIR, DATASET, LANG, counts=COUNTS, subset=SUBSET, seed=42, k=3)
print(corpus['audit'])
'''),
md('''## 6. Fine-tuning SBERT mới

Chỉ sinh cặp từ train; validation phục vụ lựa chọn checkpoint. Mỗi bài đóng góp tối đa
8 cặp trải đều theo điểm tương đồng từ vựng; nhãn liên tục là ROUGE-1 giữa câu nguồn
và câu tham chiếu phù hợp nhất. Đây là weak supervision, không phải nhãn ngữ nghĩa do người gán.

CosineSimilarityLoss tối thiểu hóa sai số bình phương giữa cosine embedding và nhãn.
Dùng learning rate 2e-5, weight decay 0,01, warmup 10%, 3 epoch.
Các giá trị là thiết lập thực nghiệm, không khẳng định tối ưu sẵn.
Checkpoint được chọn bằng độ tương quan trên các cặp validation; khác với lựa chọn
bộ phân loại câu bằng ROUGE-L ở phần sau. Chỉ fine-tune encoder một seed;
độ lệch chuẩn các seed sau đó không đo bất định của quá trình fine-tuning này.
'''),
code('''finetuned_path = finetune_encoder(corpus, RUN_DIR, BASE_MODEL,
    epochs=FT_EPOCHS, batch_size=32, learning_rate=2e-5, weight_decay=0.01, seed=42)
print('Checkpoint mới:', finetuned_path)
'''),
code('''# 7. Embedding của cùng các bài, hai encoder; không dùng API và không fallback mô hình.
pretrained_embeddings = cache_embeddings(corpus, RUN_DIR, BASE_MODEL, tag='pretrained')
finetuned_embeddings = cache_embeddings(corpus, RUN_DIR, finetuned_path, tag='finetuned')
'''),
md('''## 8. Huấn luyện bộ chọn câu và lựa chọn mô hình

Nhãn câu oracle xấp xỉ được tạo bằng greedy tối đa ROUGE-1 + ROUGE-2 với ngân sách 3 câu.
Cả Logistic và MLP dùng embedding pretrained đóng băng, vị trí chuẩn hóa, log độ dài
và cosine với tâm bài. StandardScaler chỉ fit train. Không đưa reference vào đặc trưng.

Logistic: negative log-likelihood Bernoulli với L2, C ∈ {0,01; 1; 100}, solver L-BFGS.
MLP: một lớp ẩn 128 ReLU, cross-entropy với L2 alpha ∈ {0; 0,001; 0,1}, Adam/SGD,
cùng learning rate khởi tạo 0,001. SGD dùng momentum mặc định của sklearn;
đây là so sánh hai cấu hình optimizer tại learning rate cố định, không kết luận optimizer
nào tốt nhất với mọi learning rate. Không dùng dropout trong MLP sklearn này.

Early stopping MLP theo validation log-loss, patience 5, tối đa 40 epoch.
Chọn siêu tham số theo validation ROUGE-L, seed 42; sau đó lặp cấu hình thắng với
3 seed và 25%/50%/100% số bài train. Learning curve dùng các tập con lồng nhau.
Không dùng sentence-level random validation nội bộ. Các phép thử trên cùng validation
có thể gây thiên lệch lựa chọn; đây không phải nested cross-validation.
'''),
code('''selection = run_training(corpus, pretrained_embeddings, RUN_DIR,
    seeds=SEEDS, fractions=FRACTIONS, max_epochs=MAX_EPOCHS)
print('Cấu hình đã khóa:', {key: value['config'] for key, value in selection['winners'].items()})
'''),
md('''## 9. Đánh giá test sau khi khóa lựa chọn

Tất cả phương pháp chọn đúng 3 câu từ cùng tập câu ứng viên và khôi phục thứ tự gốc.
Lead-3, TF-IDF TextRank, centroid, K-Means thuần, Logistic, MLP. So sánh pretrained
với fine-tuned dùng cùng centroid/K-Means: không gộp thay đổi encoder với thay đổi bộ chọn câu.
K-Means thuần là ablation có kiểm soát, chưa bao gồm adaptive K/position weighting/
lọc trùng của sản phẩm cũ. Các mô hình đều không dùng post-filter trong bảng chính.

ROUGE tiếng Việt dùng token âm tiết Unicode có dấu; tiếng Anh dùng stemming mặc định.
ROUGE-L ở đây là LCS toàn văn bản, không phải ROUGE-Lsum.
Ngân sách theo câu chưa bảo đảm bằng nhau về số từ; bảng lưu thêm độ dài và tỷ lệ giữ từ.

Xuất điểm từng bài, trung bình từng seed và độ lệch chuẩn giữa seed.
Paired bootstrap 2.000 lần theo bài báo cho chênh lệch ROUGE-L với pretrained K-Means,
sau khi trung bình các seed của từng bài. CI 95% chưa hiệu chỉnh nhiều so sánh,
mang tính khám phá; CI này không đồng thời đo bất định do chọn siêu tham số/fine-tuning.
Thời gian selector tách khỏi thời gian embedding; không gọi đó là độ trễ end-to-end.
'''),
code('''test_rows = run_test(corpus, pretrained_embeddings, selection, RUN_DIR,
                     finetuned_embeddings=finetuned_embeddings)
'''),
code('''# 10. BERTScore độc lập: có thể chạy lại nếu Colab ngắt, đã có cache theo bản tóm tắt.
# Không trả 0 khi lỗi tải model. Cell lỗi phải xử lý rồi mới dùng bảng có BERTScore.
add_bertscore(corpus, RUN_DIR, batch_size=16)
'''),
code('''# 11. Xuất hình và xem bảng cuối.
make_plots(RUN_DIR)
import pandas as pd
from IPython.display import display, Image
display(pd.read_csv(RUN_DIR / 'test_summary.csv'))
display(pd.read_csv(RUN_DIR / 'paired_bootstrap.csv'))
display(Image(filename=str(RUN_DIR / 'figures/learning_curve.png')))
display(Image(filename=str(RUN_DIR / 'figures/regularization_optimizer.png')))
display(Image(filename=str(RUN_DIR / 'figures/mlp_convergence.png')))
'''),
md('''## 12. Lưu kết quả để cập nhật báo cáo

Các bảng CSV, hình, cấu hình và checkpoint đã được lưu trong thư mục kết quả
trên Drive. Lưu notebook có output sau khi chạy xong và gửi lại cùng các bảng kết quả
để cập nhật những phần cần thiết trong báo cáo cũ. Giữ nguyên font, cỡ chữ,
văn phong và cách trình bày của báo cáo.
'''),
code('''from datetime import datetime, timezone
save_json(RUN_DIR / 'execution_status.json', dict(
    finished_utc=datetime.now(timezone.utc).isoformat(),
    ready_for_report=(RUN_DIR / 'bertscore_metadata.json').exists()))
print('Hoàn tất. Kết quả được lưu tại:', RUN_DIR)
print('Các bảng cần gửi lại:')
for name in ('test_summary.csv', 'test_per_document.csv', 'paired_bootstrap.csv',
             'validation_experiments.csv'):
    print(' -', RUN_DIR / name)
'''),
]

notebook = dict(nbformat=4, nbformat_minor=5, metadata={
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python'}, 'accelerator': 'GPU',
    'colab': {'name': 'AdvancedML_ExtractiveSummarizer.ipynb', 'provenance': []},
}, cells=cells)
for i, cell in enumerate(cells):
    cell['id'] = f'aml-{i:03d}'
(ROOT/'AdvancedML_ExtractiveSummarizer.ipynb').write_text(
    json.dumps(notebook, ensure_ascii=False, indent=1)+'\n', encoding='utf-8')
print(f'Đã tạo notebook end-to-end {len(cells)} cell.')
