import os
import warnings
import logging
import datasets
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses
from src.config import MODEL_CONFIGS, MODELS_DIR
from src.dataset import load_evaluation_dataset, generate_oracle_extractive_pairs
warnings.filterwarnings("ignore")
datasets.logging.set_verbosity_error()
logging.getLogger("datasets").setLevel(logging.ERROR)


def train_finetune_sbert(lang: str = 'vi', epochs: int = 3, batch_size: int = 32, sample_data_count: int = 500):
    """
    Fine-tune mô hình SBERT trên các cặp câu Oracle với CosineSimilarityLoss.
    """
    
    # 1. Load Pretrained SBERT Model
    base_model_name = MODEL_CONFIGS[lang]
    print(f"Nạp mô hình SBERT Gốc: {base_model_name}")
    model = SentenceTransformer(base_model_name)

    # 2. Nạp dữ liệu và sinh cặp câu Oracle
    raw_articles = load_evaluation_dataset(lang=lang, sample_count=sample_data_count)
    if not raw_articles:
        print(f"Không tìm thấy bài báo huấn luyện. Bỏ qua fine-tuning.")
        return
    # Tự động điều chỉnh số lượng cặp câu Oracle theo sample_data_count
    max_oracle_pairs = sample_data_count * 4
    train_examples = generate_oracle_extractive_pairs(raw_articles, max_pairs=max_oracle_pairs)
    if not train_examples:
        print("Không tạo được cặp câu huấn luyện. Bỏ qua fine-tuning.")
        return
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

    # 3. Định nghĩa Hàm Loss
    train_loss = losses.CosineSimilarityLoss(model=model)

    # 4. Thư mục lưu
    save_path = os.path.join(MODELS_DIR, f"finetuned_sbert_{lang}")
    os.makedirs(save_path, exist_ok=True)

    # 5. Huấn luyện mô hình
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=100,
        output_path=save_path,
        show_progress_bar=True
    )

    print(f"Fine-Tuning hoàn tất. Mô hình đã được lưu tại: {save_path}\n")
    return save_path


if __name__ == "__main__":
    train_finetune_sbert(lang='en', epochs=3)
