# Thực nghiệm Học máy nâng cao

Notebook duy nhất cần chạy: `AdvancedML_ExtractiveSummarizer.ipynb`.
Các bước của bài NLP được cập nhật tại vị trí phù hợp trong một luồng:
chuẩn bị dữ liệu → fine-tuning → embedding → huấn luyện Logistic/MLP →
lựa chọn mô hình → đánh giá → bảng và hình. Không có phần code phụ lục hay chế độ chạy riêng.
Notebook NLP gốc vẫn giữ nguyên output để tham khảo; không cần mở để chạy bài mới.

## Chạy trên Colab

1. Mở notebook từ repository: https://github.com/kttt294/Extractive-Summarizer
2. Chọn GPU T4, chạy từ đầu đến cuối bằng Run all.
3. Cell đầu tự clone/cập nhật mã; không cần kết nối Google Drive.
4. Cell cuối nén toàn bộ kết quả và mô hình thành `advanced_ml_v2_vi_results.zip`
   rồi tự tải về máy. Lưu thêm notebook có output.
5. Gửi lại notebook có output và ZIP kết quả để kiểm tra và sửa báo cáo.

Không cần tải lên hay giải nén gói mã nguồn. Cấu hình mặc định chạy nghiên cứu tiếng Việt.
Nếu cần chạy thêm tiếng Anh, đổi `LANG = 'en'` và chạy lại.

Kết quả lưu tại `outputs/advanced_ml/advanced_ml_v2_vi` trong thư mục repository
của phiên Colab. ZIP chứa toàn bộ thư mục này, bao gồm mô hình và embedding.
Tải file về trước khi kết thúc phiên; nếu runtime bị xóa thì dữ liệu chưa tải sẽ mất.
Nếu chạy lại trong cùng runtime, các bước hoàn tất được đọc từ cache cục bộ.
Fine-tuning đang dở sẽ chạy lại bước đó. Nếu đã đổi mã, phiên bản thư viện
hoặc giao thức thực nghiệm, đặt `RUN_NAME` mới để tránh trộn kết quả cũ.
Không lựa chọn lại siêu tham số theo điểm test.

## Thiết kế thực nghiệm

- Chính: VietNews, 3.000 train / 500 validation / 1.000 test; lấy mẫu shuffle cố định.
- Kiểm tra và loại trùng chính xác sau chuẩn hóa, chưa kiểm tra gần trùng.
- Cùng câu ứng viên và ngân sách 3 câu cho tất cả phương pháp. Báo cáo thêm độ dài
  từ vì cùng số câu chưa đồng nghĩa cùng số từ.
- Fine-tune encoder từ train mới, checkpoint chọn bằng validation MSE thấp nhất.
  Training loss ghi tối đa mỗi 50 bước; validation MSE khoảng bốn lần mỗi epoch,
  có thêm mốc trước train. Lưu CSV/JSON và hiển thị `figures/sbert_loss_curve.png`
  ngay sau fine-tuning. Đọc phần hướng dẫn phân tích dưới hình trong notebook.
- So sánh pretrained/fine-tuned với centroid/K-Means thuần, không thêm heuristic
  sản phẩm chỉ ở một phía. Fine-tuning một seed; seed K-Means không thay thế seed
  fine-tuning. Chưa có kết quả ngoài miền trong giao thức chính mới.
- Logistic/MLP dùng cùng embedding pretrained đóng băng và ba đặc trưng phụ:
  vị trí, log độ dài, cosine với tâm bài. Scaler fit train.
- Nhãn chọn câu là greedy ROUGE-1+2 oracle xấp xỉ, không phải người gán nhãn.
- Logistic L2: C=0,01/1/100. MLP một lớp 128 ReLU: L2 alpha=0/0,001/0,1,
  Adam/SGD cùng lr=0,001; SGD dùng momentum mặc định sklearn, không phải SGD thuần.
  So sánh này có điều kiện theo lr cố định, không khẳng định tối ưu toàn cục.
- Early stopping MLP theo validation log-loss, chọn cấu hình theo validation ROUGE-L
  ở seed 42; lặp các cấu hình thắng với seed 42/123/2026 và 25%/50%/100% train.
  Không dùng random validation theo câu nội bộ, không phải nested CV.
- Test chỉ chạy sau khi khóa lựa chọn. Xuất điểm từng bài, std giữa seed và paired
  bootstrap CI theo bài sau seed-averaging. CI khám phá chưa chỉnh nhiều so sánh.
- BERTScore dùng encoder đánh giá riêng, không dùng model đang fine-tune. Văn bản
  dài có thể bị scorer cắt ngắn. Không giả lập điểm 0 khi lỗi tải/đánh giá.

## Đánh giá và báo cáo

ROUGE tiếng Việt mới giữ ký tự có dấu bằng tokenizer Unicode NFC theo âm tiết.
Do đó không so trực tiếp trị số với output tiếng Việt cũ. Tiếng Anh giữ stemming.
Nguồn triển khai mặc định: https://github.com/google-research/google-research/blob/master/rouge/tokenize.py

Bảng chính mới dùng cùng ba câu cho các phương pháp; không dùng bảng cũ với ngân sách
độ dài khác nhau. BERTScore được tính độc lập, lỗi không bị thay bằng điểm 0.

Báo cáo `docs/BaoCao_BTL.docx` chưa sửa. Sau khi có kết quả, chỉ sửa những phần cần
về mục tiêu, lý thuyết, phương pháp, thực nghiệm và kết luận; giữ font, cỡ chữ,
văn phong và cách trình bày cũ.

## Kiểm thử dành cho phát triển mã

`python -m pytest -q tests/test_advanced_ml.py`

Kiểm thử dữ liệu tổng hợp kiểm tra Unicode, scaler, tách train/test, huấn luyện
Logistic/MLP, bootstrap, xuất CSV/hình và thứ tự notebook. BERTScore dùng mock
trong kiểm thử; lượt mô hình thật chạy trên Colab. Notebook không yêu cầu chạy kiểm thử.
