# Extractive Summarizer AI: SBERT Fine-Tuned + K-Means

Mô hình tóm tắt văn bản trích xuất (Extractive Summarization) song ngữ Anh - Việt ứng dụng **Sentence-BERT Fine-Tuned**, **K-Means Clustering**, **Post-filtering (Lọc trùng Cosine)** và **Khung Đánh giá Kép (Intrinsic & Extrinsic Metrics)**.

---

## Tính năng Chính

1. **Khung Đánh giá Kép (Dual-Evaluation Framework):**
   - **Nội tại (Intrinsic Metrics):** Silhouette Score (độ phân tách cụm), Diversity Score (độ đa dạng ngữ nghĩa).
   - **Ngoại tại (Extrinsic Metrics):** ROUGE-1, ROUGE-2, ROUGE-L, BERTScore F1, G-Eval.
2. **Fine-Tuning SBERT (Bài toán Regression with CosineSimilarityLoss):**
   - Giải quyết triệt để góp ý của Giảng viên ở buổi Giữa kỳ.
   - Tăng ROUGE-1 từ $38.2\%$ lên $\mathbf{42.5\%}$ ($+4.3\%$).
3. **Sản phẩm Web App Full-Stack (FastAPI + React + TailwindCSS):**
   - Tự động nhận diện ngôn ngữ bài viết (`langdetect`).
   - Tóm tắt qua đoạn văn thô, đường link URL bài báo (`newspaper3k`), hoặc theo từ khóa chủ đề.
   - Tính năng **Interactive Highlighting**: Di chuột vào câu tóm tắt thì câu tương ứng trong đoạn gốc tự động phát sáng.
4. **Triển khai Đóng gói Docker Compose:**
   - 1 câu lệnh deploy toàn bộ Frontend & Backend lên VPS.

---

## Hướng dẫn Chạy Thử nghiệm (Colab / Local)

### 1. Thử nghiệm Thuật toán & Fine-Tuning (Notebook)

Mở file `run_experiments.ipynb` trên Google Colab hoặc Jupyter Notebook local:

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
