# Cập nhật website v2 trên VPS bằng Docker Compose

Tiếng Việt dùng SBERT pretrained + Logistic Regression, C=0,01, seed 42. Encoder,
revision, checksum head và thứ tự đặc trưng được cố định theo notebook đã chạy.
Không dùng encoder fine-tuned thay cho encoder pretrained của bộ phân loại này.
Tiếng Anh giữ SBERT + K-Means, hiển thị rõ là phương pháp kế thừa.

## Cập nhật tại thư mục dự án trên VPS

```bash
git pull --ff-only origin main
cp .env.example .env
nano .env
chmod 600 .env
```

Nếu đã có `.env`, giữ file hiện có và bổ sung các biến thiếu, không copy đè.
Điền `HF_TOKEN` bằng token có quyền **read** với repository riêng tư
`kttt294/vietnamese-extractive-heads-advanced-ml-v2`. Không ghi token vào Dockerfile,
GitHub, mã frontend hoặc lệnh build. Chỉ backend nhận token khi container chạy.

```dotenv
HF_TOKEN=<token của bạn>
MODEL_CPU_THREADS=2
VI_HEAD_PATH=
```

```bash
docker compose config --quiet
docker compose up -d --build
docker compose logs -f --tail=100 backend
```

Chờ log `Vietnamese Logistic v2 ready.`. Lần đầu backend tải encoder khoảng 540 MB
và head nhỏ; cache lưu ở volume `hf_cache`. Frontend chờ backend healthy trước khi
khởi động. Không chạy `docker compose down -v` nếu muốn giữ cache.
Backend chạy một worker để tránh nhân bản bộ nhớ mô hình.

Nginx vẫn dùng HTTPS và chứng chỉ hiện có cho `syssummarizers.tech` tại
`/etc/letsencrypt/live/syssummarizers.tech/`. Giữ thư mục chứng chỉ trên VPS;
nếu tên miền khác, sửa `frontend/nginx.conf` trước khi build.
Cổng API 8000 chỉ bind localhost; truy cập ngoài qua Nginx `/api/`.

## Kiểm tra sau deploy

```bash
docker compose ps
curl --fail http://127.0.0.1:8000/api/v1/health
```

Health phải trả `version: 2.0.0` và `vietnamese_model_ready: true`.
Trên website, tóm tắt một bài tiếng Việt: kết quả ghi SBERT + Logistic Regression,
tối đa 3 câu, không có Silhouette Score, rê chuột tô đúng câu gốc.
Kiểm tra thêm URL/chủ đề và tiếng Anh. Khi chọn Auto, tiếng Việt vẫn chọn ba câu;
tiếng Anh dùng độ dài tiêu chuẩn. Độ dài tùy chỉnh chỉ xuất hiện khi chọn tiếng Anh.

Tham chiếu notebook chỉ áp dụng cho cùng văn bản và cùng tiền xử lý. Tính năng gom
nhiều bài theo chủ đề là mở rộng sản phẩm, chưa có đánh giá riêng trong bảng test.

Nếu model riêng tư không tải được, xem quyền read/token và kết nối mạng trong log;
backend sẽ không âm thầm chuyển sang checkpoint khác.

## Không tải head từ Hub (tùy chọn)

Copy artifact đã kiểm tra `logistic_seed42.joblib` vào thư mục `models/` trên VPS,
đặt `VI_HEAD_PATH=/app/models/logistic_seed42.joblib` trong `.env`, rồi restart backend.
Encoder công khai vẫn được tải từ revision đã cố định và giữ trong cache.
Checksum head phải đúng bản đã dùng thực nghiệm; không copy file MLP hoặc model seed khác.

## Chạy image gộp

Dockerfile ở gốc cũng được cập nhật cho cùng backend v2 và frontend mới. Chạy với
`--env-file .env` và mount cache vào `/home/user/app/.cache/huggingface`; cổng mặc định
là 10000. Đối với VPS đang chạy HTTPS, ưu tiên luồng `docker compose` phía trên.
