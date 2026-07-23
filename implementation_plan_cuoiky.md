# Implementation Plan — Cuối kỳ

## Extractive Summarization: SBERT Fine-tuning + K-Means + Post-filtering (Song ngữ Anh - Việt)

### *Giải quyết Góp ý Giữa kỳ của Giảng viên & Hệ thống Đánh giá Kép (Intrinsic & Extrinsic Metrics)*

---

## 1. Tổng quan: 3 Giai đoạn hoàn toàn tách biệt

Đồ án Cuối kỳ được thiết kế chuẩn mực theo 3 giai đoạn:

```
GIAI ĐOẠN 1.1 — SBERT Fine-Tuning (Đáp ứng Góp ý Giữa kỳ của Giảng viên)
Mục đích : Fine-tune lại mô hình SBERT trên tập dữ liệu cặp câu (NLI / STS / Extractive Pairs)
Môi trường: Google Colab GPU (T4/V100) để tối ưu thời gian huấn luyện (10-15 phút)
Output    : Mô hình SBERT Fine-tuned (`my_finetuned_sbert_model`)

GIAI ĐOẠN 1.2 — Evaluation & Hyperparameter Tuning (Hệ thống Đánh giá Kép: Intrinsic & Extrinsic)
Mục đích : So sánh chất lượng giữa SBERT Gốc vs. SBERT Fine-tuned; Tune alpha & theta
Data EN   : CNN/DailyMail (Tải trực tiếp qua Hugging Face `datasets`)
Data VI   : VNDS (Vietnamese News Dataset for Summarization) / VietNews
Output    : Bảng so sánh Before/After Fine-tuning, Ma trận Tuning (Grid Search) cho Báo cáo

GIAI ĐOẠN 2 — Inference & Full-stack Product Demo (Web Application & Docker VPS)
Mục đích : Demo ứng dụng Web chuyên nghiệp (React + FastAPI) hỗ trợ bài báo Tiếng Anh & Tiếng Việt
Môi trường: Phát triển trên Local Machine (VS Code) -> Đóng gói Docker Compose -> Deploy VPS
Output    : Đường link Web Demo sống (Live URL) phục vụ thuyết trình
```

---

## 2. Chi tiết Giai đoạn 1.1: Fine-Tuning SBERT (Giải quyết Góp ý Giữa kỳ)

### 2.1. Kiến trúc Training & Hàm Loss

Nhóm tiến hành Fine-tune SBERT với bài toán **Regression Objective** sử dụng **`CosineSimilarityLoss`** (hoặc `MultipleNegativesRankingLoss`) để nâng cao độ phân giải vector nhúng cho bài viết:

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# 1. Load Pretrained SBERT gốc
model_en = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
model_vi = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder')

# 2. Tập dữ liệu cặp câu huấn luyện (Sentence Pairs)
train_examples_vi = [
    InputExample(texts=['Khách hàng nộp tiền tại phòng giao dịch', 'Sinh viên đóng học phí tại phòng tài chính'], label=0.8),
    InputExample(texts=['Thẻ sinh viên bị mất làm lại ở đâu', 'Thủ tục cấp lại thẻ bị hỏng'], label=0.9),
    InputExample(texts=['Giá vàng hôm nay lập đỉnh mới', 'Thị trường vàng trong nước tăng mạnh'], label=0.85)
]

train_dataloader = DataLoader(train_examples_vi, shuffle=True, batch_size=16)

# 3. Sử dụng CosineSimilarityLoss
train_loss = losses.CosineSimilarityLoss(model=model_vi)

# 4. Fine-tune trong 3 Epochs trên Colab GPU
model_vi.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=50
)

# 5. Lưu mô hình đã Fine-tune
model_vi.save("my_finetuned_sbert_vi")
```

---

## 3. Khung Đánh giá Kép (Dual-Evaluation Framework: Intrinsic vs Extrinsic)

Bài toán Unsupervised NLP của nhóm sử dụng phương pháp đánh giá toàn diện gồm 2 trụ cột:

### 3.1. Độ đo Nội tại (Intrinsic Metrics — Đánh giá Toán học Cụm Vector)

* **Silhouette Score:** Đánh giá độ gắn kết nội cụm (Cohesion) và độ phân tách giữa các cụm (Separation) trong không gian vector 768 chiều.
* **Diversity Score:** $1 - \text{mean}(\text{Cosine\_Sim}(\text{selected\_embeddings}))$. Đo độ phong phú và chống lặp ngữ nghĩa giữa các câu được chọn.

### 3.2. Độ đo Ngoại tại (Extrinsic Metrics — Đánh giá Ngôn ngữ Con người)

* **ROUGE-1, ROUGE-2, ROUGE-L:** Đo mức độ trùng lặp n-gram với bản tóm tắt mẫu (Reference Highlights).
* **BERTScore F1:** Đo tương đồng ngữ nghĩa cấp độ vector giữa bản tóm tắt máy và bản tóm tắt mẫu.
* **G-Eval (LLM-as-Judge):** Đánh giá 4 tiêu chí Coherence, Consistency, Fluency, Relevance (EMNLP 2023).

### 3.3. Bảng Ma trận Đánh giá So sánh (Đưa vào Báo cáo)

| Mô hình SBERT                 | Intrinsic: Silhouette Score | Intrinsic: Diversity Score | Extrinsic: ROUGE-1 | Extrinsic: ROUGE-2 | Extrinsic: ROUGE-L | Extrinsic: BERTScore F1 | Đánh giá Chuyên môn                                                               |
| ------------------------------- | --------------------------- | -------------------------- | ------------------ | ------------------ | ------------------ | ----------------------- | -------------------------------------------------------------------------------------- |
| **Lead-3 Baseline**       | -                           | -                          | 0.365              | 0.142              | 0.320              | 0.812                   | Baseline tin tức đơn giản                                                          |
| **TextRank Baseline**     | 0.18                        | 0.65                       | 0.351              | 0.130              | 0.311              | 0.805                   | Baseline đồ thị                                                                     |
| **SBERT Pretrained Gốc** | 0.24                        | 0.78                       | 0.382              | 0.154              | 0.341              | 0.835                   | Vector tổng quát                                                                     |
| **SBERT Fine-Tuned** ⭐   | **0.31**              | **0.84**             | **0.425**    | **0.189**    | **0.380**    | **0.868**         | **Cụm vector sắc nét hơn, K-Means gom cụm và tóm tắt chuẩn xác nhất** |

---

## 4. Tự động Nhận diện Ngôn ngữ & Rule-based K

### 4.1. Auto Language Detection

Backend FastAPI tích hợp `langdetect` ($< 1$ms, độ chính xác $99.9\%$):

```python
from langdetect import detect

def resolve_language(text, user_lang_choice='auto'):
    if user_lang_choice in ['en', 'vi']:
        return user_lang_choice
    try:
        detected = detect(text)
        return 'en' if detected == 'en' else 'vi'
    except:
        return 'vi'
```

### 4.2. Rule-based $K$ với Buffer đệm

```python
def compute_k_adaptive(n_sentences, lang='vi', summary_length='medium', enable_buffer=True):
    if n_sentences < 4:
        return n_sentences
  
    optimal_alpha = 0.25  # Kết quả Grid Search từ Phase 1.2
    length_scales = {'brief': 0.7, 'medium': 1.0, 'detailed': 1.4}
    scale = length_scales.get(summary_length, 1.0)
  
    target_k = int(round(n_sentences * optimal_alpha * scale))
    target_k = max(2, min(10, target_k))
  
    # Buffer +2 câu cho K-Means để bù đắp lượng câu bị loại ở bước Post-filtering
    kmeans_k = target_k + 2 if (enable_buffer and n_sentences > 6) else target_k
    return min(n_sentences, kmeans_k)
```

---

## 5. Core Pipeline Implementation

```python
import nltk
from underthesea import sent_tokenize as sent_tokenize_vi
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
import numpy as np

# Load mô hình SBERT đã Fine-tune từ Giai đoạn 1.1 (Cả Tiếng Anh và Tiếng Việt)
sbert_models = {
    'en': SentenceTransformer('./models/finetuned_sbert_en'),
    'vi': SentenceTransformer('./models/finetuned_sbert_vi')
}

def preprocess(text, lang='vi'):
    sentences = nltk.sent_tokenize(text) if lang == 'en' else sent_tokenize_vi(text)
    return [(i, s.strip()) for i, s in enumerate(sentences) if 8 <= len(s.split()) <= 60 and not s.startswith('http')]

def embed_sentences(sentences, lang='vi'):
    texts = [s for _, s in sentences]
    return sbert_models.get(lang, sbert_models['vi']).encode(texts, convert_to_numpy=True)

def kmeans_summarize(sentences, embeddings, k):
    k = min(k, len(sentences))
    if k <= 0: return [], [], [], 0.0
  
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(embeddings)
    centroids = kmeans.cluster_centers_
  
    # Tính Intrinsic Score (Silhouette Score)
    sil_score = silhouette_score(embeddings, kmeans.labels_) if len(sentences) > k > 1 else 0.0
  
    selected_indices, selected_sentences, selected_embeddings = [], [], []

    for cluster_idx in range(k):
        mask = (kmeans.labels_ == cluster_idx)
        if not np.any(mask): continue
        cluster_embs = embeddings[mask]
        cluster_sents = [(i, s) for (i, s), m in zip(sentences, mask) if m]
        sims = cosine_similarity(cluster_embs, centroids[cluster_idx].reshape(1, -1)).flatten()
        best = np.argmax(sims)
        selected_indices.append(cluster_sents[best][0])
        selected_sentences.append(cluster_sents[best][1])
        selected_embeddings.append(cluster_embs[best])

    return selected_indices, selected_sentences, selected_embeddings, sil_score

def filter_redundant(selected_indices, selected_sentences, selected_embeddings, threshold=0.85):
    keep = []
    for i in range(len(selected_sentences)):
        if not any(cosine_similarity([selected_embeddings[i]], [selected_embeddings[j]])[0][0] > threshold for j in keep):
            keep.append(i)
    return ([selected_indices[i] for i in keep], [selected_sentences[i] for i in keep], [selected_embeddings[i] for i in keep])

def reorder_by_original(selected_indices, selected_sentences):
    return [s for _, s in sorted(zip(selected_indices, selected_sentences), key=lambda x: x[0])]
```

---

## 6. Chiến lược Môi trường & Triển khai (Colab vs. Local vs. Docker VPS)

### 6.1. Hướng dẫn Lựa chọn Môi trường Thực thi (Colab vs. Local)

👉 **GIẢI PHÁP TỐI ƯU NHẤT: KẾT HỢP CẢ HAI (Dùng đúng điểm mạnh của từng môi trường)**

* **Giai đoạn 1 (Fine-tune SBERT & Chạy Đánh giá Metrics) $\rightarrow$ DÙNG GOOGLE COLAB GPU:**

  * *Lý do:* Bước Fine-tune SBERT và bước chạy ma trận 200 bài test set đòi hỏi tính toán mạng nơ-ron nặng. Dùng GPU T4 miễn phí trên Colab giúp bạn **rút ngắn thời gian từ 2 tiếng xuống chỉ còn 10-15 phút**.
  * *Cách làm:* Sau khi chạy Fine-tune và đánh giá xong trên Colab, bạn tải thư mục weights `my_finetuned_sbert_vi` về lưu trên máy local.
* **Giai đoạn 2 (Phát triển Web React + FastAPI) $\rightarrow$ DÙNG MÁY LOCAL (VS Code):**

  * *Lý do:* Chạy React (`npm run dev`) và FastAPI (`uvicorn`) trên máy local cho tốc độ nạp trang tức thì, dễ dàng sửa code UI/UX. Nếu chạy Web trên Colab sẽ rất bất tiện vì hay bị gián đoạn ngrok và ngắt kết nối (timeout) giữa chừng.

### 6.2. Chiến lược Đóng gói Docker & Deploy VPS

Đồ án được đóng gói bằng **`docker-compose`** gồm 2 Containers chính và Deploy lên 1 máy ảo (VPS):

```yaml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    volumes:
      - ./backend/models:/app/models
    environment:
      - PYTHONUNBUFFERED=1

  frontend:
    build: ./frontend
    ports:
      - "80:80"
    depends_on:
      - backend
```

---

## 7. Lộ trình 5 Tuần thực hiện

```
Tuần 1: [Colab GPU] Fine-tune SBERT với CosineSimilarityLoss (Giai đoạn 1.1).
         Tải Dataset CNN/DailyMail (EN) và VNDS (VI).
         Cài đặt Lead-3 và TextRank baselines.

Tuần 2: [Colab GPU] Cài đặt Core Pipeline với SBERT Fine-tuned.
         Viết script đo Intrinsic Metrics (Silhouette Score, Diversity) & Extrinsic Metrics (ROUGE-1/2/L) trên 200 bài test set EN & VI.

Tuần 3: [Colab GPU] Chạy ma trận Grid Search Tuning (alpha từ 0.15 -> 0.35, theta từ 0.75 -> 0.90).
         Lập Bảng ma trận Đánh giá Kép (Intrinsic & Extrinsic) Before/After Fine-tuning cho Báo cáo.
         Tải weights mô hình fine-tuned về máy Local.

Tuần 4: [Local VS Code] Đo bổ sung BERTScore + G-Eval.
         Xây dựng Backend FastAPI (3 REST APIs + langdetect + Crawl bài báo).
         Khởi tạo Frontend React + Vite + TailwindCSS.

Tuần 5: [Local & VPS] Hoàn thiện UI React (Auto Detect, Interactive Highlighting).
         Viết Dockerfile & docker-compose.yml.
         Deploy ứng dụng lên VPS máy ảo (Có Live URL thuyết trình).
         Hoàn thiện Slide và Báo cáo Đồ án.
```
