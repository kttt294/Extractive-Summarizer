import os
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses
from src.config import MODEL_CONFIGS, MODELS_DIR
from src.dataset import load_evaluation_dataset, generate_oracle_extractive_pairs


def train_finetune_sbert(lang: str = 'vi', epochs: int = 2, batch_size: int = 16, sample_data_count: int = 200):
    """
    Fine-tunes SBERT model on Oracle Extractive Pairs using CosineSimilarityLoss.
    Satisfies Professor's Midterm Feedback!
    """
    print(f"\n==========================================")
    print(f"  STARTING SBERT FINE-TUNING ({lang.upper()})  ")
    print(f"==========================================")

    # 1. Load Pretrained SBERT Model
    base_model_name = MODEL_CONFIGS[lang]
    print(f"Loading Base SBERT Model: {base_model_name}")
    model = SentenceTransformer(base_model_name)

    # 2. Load Data & Generate Oracle Pairs
    raw_articles = load_evaluation_dataset(lang=lang, sample_count=sample_data_count)
    if not raw_articles:
        print(f"No training articles found for {lang}. Skipping fine-tuning.")
        return

    train_examples = generate_oracle_extractive_pairs(raw_articles, max_pairs=1000)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

    # 3. Define Objective & Loss Function
    train_loss = losses.CosineSimilarityLoss(model=model)

    # 4. Save Directory
    save_path = os.path.join(MODELS_DIR, f"finetuned_sbert_{lang}")
    os.makedirs(save_path, exist_ok=True)

    # 5. Train Model
    print(f"Training SBERT for {epochs} epochs...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=100,
        output_path=save_path,
        show_progress_bar=True
    )

    print(f"Fine-Tuning complete! Model saved to: {save_path}\n")
    return save_path


if __name__ == "__main__":
    train_finetune_sbert(lang='en', epochs=2)
