# Hệ thống Tóm tắt Văn bản Đa lĩnh vực (Extractive Summarization AI)

Mô hình tóm tắt văn bản trích xuất (Extractive Summarization) song ngữ Anh - Việt ứng dụng **Sentence-BERT Fine-Tuned**, **K-Means Clustering**, **Dynamic Adaptive K**, **Post-filtering (Lọc trùng Cosine)** và **Khung Đánh giá Kép (Intrinsic & Extrinsic Metrics)**.

---

## Tính năng Chính

1. **Khung Đánh giá Kép (Dual-Evaluation Framework):**
   - **Nội tại (Intrinsic Metrics):** Silhouette Score (độ sắc nét phân cụm K-Means), Diversity Score (khả năng chống lặp ý), Compression Ratio (tỷ lệ nén độ dài văn bản).
   - **Ngoại tại (Extrinsic Metrics):** ROUGE-1, ROUGE-2, ROUGE-L (độ phủ từ vựng), BERTScore F1 và SBERT Cosine (độ tương đồng ngữ nghĩa so với nhãn chuẩn).
2. **Fine-Tuning SBERT & Thuật toán Dynamic Adaptive K:**
   - Tối ưu hóa mô hình SBERT với hàm mất mát CosineSimilarityLoss.
   - Tính toán số cụm $ và số câu mục tiêu thích ứng toán học hoàn toàn  = \text{round}(N \times \alpha \times \text{scale})$ theo 3 chế độ độ dài (Ngắn gọn, Tiêu chuẩn, Chi tiết).
3. **Sản phẩm Web App Full-Stack (FastAPI + React + TailwindCSS):**
   - Tự động nhận diện ngôn ngữ bài viết (langdetect).
   - Cào dữ liệu theo định hướng vùng địa lý (n-vi, us-en) khi tìm kiếm tin tức theo từ khóa chủ đề.
   - Tóm tắt qua đoạn văn thô, đường link URL bài báo (
ewspaper3k + bóc tách Sapo), hoặc theo từ khóa chủ đề tin tức 24h.
   - Tính năng **Interactive Highlighting**: Di chuột vào câu tóm tắt thì câu tương ứng trong đoạn gốc tự động phát sáng.
4. **Triển khai Đóng gói Docker Compose:**
   - 1 câu lệnh deploy toàn bộ Frontend & Backend lên VPS.

---

## Cấu trúc Mã nguồn (Project Structure)

- **NLP_ExtractiveSummarizer.ipynb**: Notebook chính dùng trên Google Colab để huấn luyện (Fine-tune) mô hình SBERT và chạy thử nghiệm đánh giá (Ablation Study).
- **src/**: Chứa các script Python cốt lõi của thuật toán tóm tắt (tiền xử lý, nhúng vector, phân cụm K-Means, tạo bộ dữ liệu huấn luyện, và các chỉ số đo lường).
- **ackend/**: Mã nguồn API Server viết bằng FastAPI (Python) dùng để phục vụ mô hình học máy ra môi trường web.
- **rontend/**: Mã nguồn Giao diện người dùng Web UI viết bằng ReactJS và TailwindCSS.
- **models/**: Thư mục lưu trữ các file trọng số (weights) sau khi mô hình được huấn luyện xong (người dùng sẽ tự tải vào đây).
- **docs/**: Chứa các tài liệu, cấu trúc báo cáo hoặc sơ đồ hệ thống tham khảo.
- **docker-compose.yml**: Tệp cấu hình để triển khai toàn bộ hệ thống (Frontend + Backend) bằng Docker chỉ với 1 câu lệnh.
- **
equirements.txt**: Danh sách các thư viện Python cần thiết cho hệ thống Backend.

---

## Hướng dẫn Chạy Thử nghiệm

### 1. Chuẩn bị dữ liệu và Huấn luyện (Google Colab)

**LƯU Ý QUAN TRỌNG TRƯỚC KHI CHẠY COLAB:** 
File Notebook NLP_ExtractiveSummarizer.ipynb được thiết kế để tự động tải mã nguồn này về môi trường Colab. Do đó, bạn **BẮT BUỘC** phải tải (push) toàn bộ thư mục code này lên một kho lưu trữ (Repository) trên tài khoản GitHub cá nhân của bạn trước tiên.
- **Bước 1:** Đẩy (Push) thư mục code này lên tài khoản GitHub của bạn.
- **Bước 2:** Mở file NLP_ExtractiveSummarizer.ipynb trên Google Colab.
- **Bước 3:** Tìm đến ô code thứ 3 (phần git clone) và thay thế đoạn link https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPOSITORY.git bằng link GitHub thực tế của bạn.
- **Bước 4:** Chạy các ô code để bắt đầu Fine-tune và Đánh giá mô hình.

### 2. Chạy Web App trên Máy Local

#### A. Backend (FastAPI):

`ash
pip install -r requirements.txt
python -m backend.main
# Server chạy tại: http://localhost:8000
# Swagger API Docs: http://localhost:8000/docs
`

#### B. Frontend (React + Vite):

`ash
cd frontend
npm install
npm run dev
# Web UI chạy tại: http://localhost:5173
`

---

## Triển khai với Docker Compose (VPS Deployment)

`ash
docker-compose up --build -d
`

- **Frontend Nginx Web UI:** http://localhost:80 (hoặc IP VPS)
- **Backend FastAPI REST API:** http://localhost:8000
