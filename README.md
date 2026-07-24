# Extractive Summarizer AI: SBERT Fine-Tuned + K-Means


**Google Colab Notebook:** [Mở Notebook Thực nghiệm trên Google Colab](https://colab.research.google.com/drive/1Y28wbSRjE1IlshfOnID0GcUP_8I_5lLz)

Mô hình tóm tắt văn bản trích xuất (Extractive Summarization) song ngữ Anh - Việt ứng dụng **Sentence-BERT Fine-Tuned**, **K-Means Clustering**, **Dynamic Adaptive K**, **Post-filtering (Lọc trùng Cosine)** và **Khung Đánh giá Kép (Intrinsic & Extrinsic Metrics)**.

---

## Tính năng Chính

1. **Khung Đánh giá Kép (Dual-Evaluation Framework):**
   - **Nội tại (Intrinsic Metrics):** Silhouette Score (độ phân tách cụm K-Means), Diversity Score (độ đa dạng ngữ nghĩa trên Không gian Tham chiếu Vector Trung tính).
   - **Ngoại tại (Extrinsic Metrics):** ROUGE-1, ROUGE-2, ROUGE-L, BERTScore F1, Compression Ratio (Tỷ lệ nén dung lượng rác).
2. **Fine-Tuning SBERT & Thuật toán Dynamic Adaptive K:**
   - Tối ưu hóa mô hình SBERT với hàm mất mát `CosineSimilarityLoss` / `MultipleNegativesRankingLoss`.
   - Tính toán số cụm $K$ và số câu mục tiêu thích ứng toán học hoàn toàn $K = \text{round}(N \times \alpha \times \text{scale})$ theo 3 chế độ độ dài (Ngắn gọn, Tiêu chuẩn, Chi tiết).
   - Đạt ROUGE-1 **52.87%** và ROUGE-L **31.03%** trên tập dữ liệu VietNews.
3. **Sản phẩm Web App Full-Stack (FastAPI + React + TailwindCSS):**
   - Tự động nhận diện ngôn ngữ bài viết (`langdetect`).
   - Cào dữ liệu theo định hướng vùng địa lý (`vn-vi`, `us-en`) khi tìm kiếm tin tức theo từ khóa chủ đề.
   - Tóm tắt qua đoạn văn thô, đường link URL bài báo (`newspaper3k` + bóc tách Sapo), hoặc theo từ khóa chủ đề tin tức 24h.
   - Tính năng **Interactive Highlighting**: Di chuột vào câu tóm tắt thì câu tương ứng trong đoạn gốc tự động phát sáng.
4. **Triển khai Đóng gói Docker Compose:**
   - 1 câu lệnh deploy toàn bộ Frontend & Backend lên VPS.

---

## Hướng dẫn Chạy Thử nghiệm (Colab / Local)

### 1. Thử nghiệm Thuật toán & Fine-Tuning (Notebook)

Mở file `Nhom2_NLP_ExtractiveSummarizer.ipynb` trên Google Colab hoặc Jupyter Notebook local:

```bash
python -m src.train
python -m src.evaluate
```

### 2. Chạy Web App trên Máy Local

#### A. Backend (FastAPI):

```bash
pip install -r requirements.txt
python -m backend.main
# Server chạy tại: http://localhost:8000
# Swagger API Docs: http://localhost:8000/docs
```

#### B. Frontend (React + Vite):

```bash
cd frontend
npm install
npm run dev
# Web UI chạy tại: http://localhost:5173
```

---

## Triển khai với Docker Compose (VPS Deployment)

```bash
docker-compose up --build -d
```

- **Frontend Nginx Web UI:** `http://localhost:80` (hoặc IP VPS)
- **Backend FastAPI REST API:** `http://localhost:8000`
