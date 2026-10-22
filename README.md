# Học biểu diễn và lựa chọn mô hình cho tóm tắt trích xuất tiếng Việt

Bài tập lớn môn Học máy nâng cao. Nghiên cứu so sánh SBERT pretrained và fine-tuned,
Logistic Regression, MLP, K-Means, centroid, Lead-3 và TextRank trên VietNews.

## Cấu trúc

- `docs/BaoCao_BTL.docx`: báo cáo hoàn chỉnh, gồm phân tích loss curve SBERT/MLP.
- `docs/AdvancedML_ExtractiveSummarizer.ipynb`: notebook end-to-end đã lưu output.
- `src/`: thuật toán; `tests/`: kiểm thử giao thức và website.
- `backend/`, `frontend/`: website demo; Dockerfile và `docker-compose.yml`
  phục vụ triển khai.
- `requirements/`: thư viện cho Colab và website; `requirements.txt` cài cả hai.

## Chạy lại thực nghiệm

Mở `AdvancedML_ExtractiveSummarizer.ipynb` trên Google Colab, chọn GPU T4 rồi Run all.
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
Tiếng Anh dùng encoder pretrained công khai với K-Means kế thừa.

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

## Thiết kế thực nghiệm

Dữ liệu VietNews được lấy mẫu với seed cố định, kiểm tra trùng chính xác sau
chuẩn hóa; chưa kiểm tra đầy đủ bài gần trùng. Mọi phương pháp cùng chọn ba câu,
nhưng độ dài theo từ vẫn khác nhau nên cần đọc kèm thống kê độ dài.

Nghiên cứu tách hai nhánh: pretrained/fine-tuned SBERT cùng centroid hoặc K-Means;
và Logistic/MLP học trên embedding pretrained đóng băng cùng ba đặc trưng phụ
(vị trí, log độ dài, cosine với tâm bài). Scaler chỉ fit train. Nhãn chọn câu
là greedy ROUGE-1+2 oracle xấp xỉ, không phải nhãn do người đọc gán.

SBERT học cosine MSE bằng AdamW trong ba epoch, seed 42; checkpoint được chọn
bằng validation MSE thấp nhất. Logistic khảo sát C = 0,01/1/100; MLP 128 ReLU
khảo sát alpha = 0/0,001/0,1 và Adam/SGD ở lr = 0,001. MLP early stopping theo
validation log-loss. Chọn siêu tham số theo validation ROUGE-L rồi lặp ba seed
42/123/2026 và các tỷ lệ train 25%/50%/100%. Đây không phải nested CV; fine-tuning
encoder chỉ chạy một seed.

Test được đánh giá sau khi khóa lựa chọn. ROUGE tiếng Việt dùng tokenizer Unicode
NFC theo âm tiết, giữ dấu; không so trực tiếp với điểm tokenizer ASCII của bài NLP
cũ. BERTScore dùng encoder đánh giá độc lập. Khoảng tin cậy paired bootstrap lấy
mẫu theo bài sau khi trung bình seed, mang tính khám phá và chưa chỉnh nhiều so sánh.

Logistic đạt ROUGE-L trung bình cao nhất trong các cấu hình đã thử, còn Lead-3
dẫn đầu ROUGE-1, ROUGE-2 và BERTScore. Validation MSE của SBERT giảm mạnh chưa
chứng minh cải thiện chắc chắn chất lượng K-Means. MLP có dấu hiệu overfitting
sau epoch 2. Xem output notebook và báo cáo để đọc biểu đồ và phân tích đầy đủ.
