import os

# Model Checkpoint Paths & Hugging Face Identifiers
MODEL_CONFIGS = {
    'en': 'sentence-transformers/all-MiniLM-L6-v2',
    'vi': 'bkai-foundation-models/vietnamese-bi-encoder',
    'finetuned_vi': './models/finetuned_sbert_vi',
    'finetuned_en': './models/finetuned_sbert_en'
}

# Optimal Hyperparameters (Obtained from Grid Search Evaluation)
OPTIMAL_HYPERPARAMS = {
    'alpha': 0.25,        # Sentence selection ratio K %
    'theta': 0.85,        # Post-filtering cosine similarity threshold
    'min_words': 8,       # Minimum sentence length in words
    'max_words': 60,      # Maximum sentence length in words
    'buffer_k': 2         # K-Means cluster count buffer for post-filtering
}

# Paths for saving models and evaluations
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
