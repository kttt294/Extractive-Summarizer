# Trích xuất thông tin quan trọng trong một bài viết: Tinh chỉnh, so sánh và lựa chọn mô hình cho tiếng Việt và tiếng Anh

Bài tập lớn môn Học máy nâng cao về tóm tắt trích xuất, tức chọn các câu quan trọng
từ văn bản gốc. Nghiên cứu tinh chỉnh và so sánh encoder SBERT trên VietNews
(tiếng Việt) và CNN/DailyMail (tiếng Anh). Hai ngôn ngữ đều có các đối chứng
Lead-3, TextRank, Centroid và K-Means. Tiếng Việt khảo sát thêm Logistic Regression và MLP.

## Cấu trúc

- [BaoCao.pdf](BaoCao.pdf): báo cáo hoàn chỉnh, gồm phân tích đường cong loss.
- `src/`: thuật toán; `tests/`: kiểm thử giao thức và website.
- `backend/`, `frontend/`: website demo; Dockerfile và `docker-compose.yml`
  phục vụ triển khai.
- `requirements/`: thư viện cho Colab và website; `requirements.txt` cài cả hai.

## Chạy lại thực nghiệm

Mở file `AdvancedML_ExtractiveSummarizer.ipynb` trên Google Colab, chọn GPU T4
rồi Run all.
Notebook tự clone mã từ https://github.com/kttt294/Extractive-Summarizer và cài
môi trường. Không cần Google Drive hoặc checkpoint đã fine-tune trước.
Cell cuối đóng gói kết quả và tải ZIP về máy. Nếu đổi mã hoặc cấu hình, dùng
`RUN_NAME` mới để tránh trộn cache; không chọn lại cấu hình theo điểm test.

Để chạy riêng ablation trên runtime mới, chỉ chạy từ **mục 13** ở cuối notebook.
Phần này tự tải head/scaler của nhóm từ Hugging Face, lấy lại đúng mẫu dữ liệu
và tạo embedding pretrained; không fine-tune lại encoder. Chọn GPU T4.
Model đã public nên không cần token hoặc đăng nhập Hugging Face.
Tám cấu hình đặc trưng được so sánh trên Logistic/MLP với ba seed; cell cuối
tải ZIP chứa các bảng, loss trace, hình và metadata, không kèm weights/cache.

Kết quả đã được trình bày trong output notebook và báo cáo. Bản nộp không kèm
thư mục kết quả rời, cache, optimizer hoặc weights. Chạy lại notebook từ đầu
trên Colab để tạo lại dữ liệu, các CSV, lịch sử loss, biểu đồ và model.

## Model đã huấn luyện

Weights lưu trên Hugging Face, không đóng kèm bài nộp:

- Encoder fine-tuned: https://huggingface.co/kttt294/vietnamese-sbert-advanced-ml-v2
- Logistic/MLP kèm scaler, ba seed: https://huggingface.co/kttt294/vietnamese-extractive-heads-advanced-ml-v2

Hai repository này đã **public**. Notebook và website tải model trực tiếp,
không cần token hoặc tài khoản Hugging Face. Không có token trong bài nộp.
Backend cố định revision và kiểm tra checksum Logistic trước khi nạp model.

Website tiếng Việt tự tải encoder pretrained công khai và Logistic seed 42 từ Hub.
Không thay encoder này bằng bản fine-tuned: head được học trên đặc trưng pretrained.
Mã backend trong repo này dùng encoder pretrained với K-Means cho tiếng Anh.
Website hỗ trợ ba chế độ cho cả nhập văn bản, URL và chủ đề, kể cả nhận diện ngôn ngữ
tự động. Tiếng Việt chọn tối đa 2/3/5 câu tương ứng Ngắn gọn/Tiêu chuẩn/Chi tiết.
Nhánh tiếng Anh tính số câu mục tiêu và số cụm K-Means theo số câu hợp lệ và chế độ
người dùng chọn. Các bảng kết quả dưới đây dùng ba câu cố định cho mỗi bài.

## Chạy website và kiểm thử

Project sử dụng **CPython 3.13**; cả hai Dockerfile đã cấu hình sẵn phiên bản này.
Khi chạy local, tạo môi trường ảo bằng Python 3.13 và kích hoạt trước khi cài thư viện
(Windows: `py -3.13 -m venv .venv`; Linux/macOS: `python3.13 -m venv .venv`).
Nếu đang có môi trường Python 3.12, tạo lại môi trường ảo thay vì dùng lại thư mục cũ.

Trên VPS, chạy `docker compose up -d --build`. Model public nên không cần tạo
`.env` hoặc khai báo `HF_TOKEN` để tải model. Cấu hình Nginx hiện dùng chứng chỉ HTTPS của
`summarizer.techvi.io.vn` tại `/etc/letsencrypt`; tạo chứng chỉ đúng tên miền trước khi chạy.

Khi phát triển local:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements/web.txt
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Mở terminal khác:

```bash
cd frontend
npm ci
npm run dev
```

Kiểm thử không tải model hoặc dataset:

```bash
pip install -r requirements/ml.txt -r requirements/web.txt pytest httpx
python -m pytest -q tests
```

Một kiểm thử đối chiếu artifact thật chỉ chạy khi có đủ weights và cache embedding;
bản nộp gọn sẽ bỏ qua kiểm thử đó. Các kiểm thử giao thức và API còn lại vẫn chạy.

## Kết quả nổi bật

Mỗi phương pháp chọn đúng ba câu trên mỗi bài để so sánh với Lead-3. Các bảng
dưới đây trích từ kết quả trên **1.000 bài test cho mỗi ngôn ngữ**. ROUGE là F1
trên thang phần trăm, BERTScore F1 trên thang 0–1.

**Tiếng Việt · VietNews**

| Phương pháp                | ROUGE-1 (%) | ROUGE-2 (%) | ROUGE-L (%) | BERTScore F1 |
| -------------------------- | ----------: | ----------: | ----------: | -----------: |
| Lead-3                     |   **26,77** |   **14,32** |       18,88 |   **0,7316** |
| SBERT pretrained + K-Means |       25,98 |       12,66 |       17,66 |       0,7273 |
| SBERT fine-tuned + K-Means |       26,13 |       12,87 |       17,86 |       0,7280 |
| Logistic Regression        |       25,95 |       13,29 |   **19,13** |       0,7227 |
| MLP                        |       25,88 |       12,97 |       18,70 |       0,7233 |

**Tiếng Anh · CNN/DailyMail**

| Phương pháp                 | ROUGE-1 (%) | ROUGE-2 (%) | ROUGE-L (%) | BERTScore F1 |
| --------------------------- | ----------: | ----------: | ----------: | -----------: |
| Lead-3                      |   **40,42** |   **17,63** |   **25,12** |   **0,8700** |
| TextRank                    |       33,37 |       12,49 |       21,41 |       0,8585 |
| SBERT pretrained + Centroid |       33,58 |       12,14 |       21,22 |       0,8593 |
| SBERT pretrained + K-Means  |       33,48 |       11,73 |       20,88 |       0,8591 |
| SBERT fine-tuned + K-Means  |       34,27 |       11,97 |       21,22 |       0,8590 |

Trong bảng tiếng Việt, Logistic Regression có ROUGE-L cao nhất.
Ở tiếng Anh, encoder fine-tuned được chọn trong hai cấu hình K-Means theo
ROUGE-L trên validation (22,31% so với 21,91%). Lead-3 dẫn đầu cả bốn chỉ số
trên test tiếng Anh. Khoảng tin cậy 95% của mức tăng ROUGE-L khi tinh chỉnh
encoder cho K-Means chứa 0 ở cả hai ngôn ngữ. Xem
[báo cáo](BaoCao.pdf) để đọc đầy đủ các bảng và phân tích.
