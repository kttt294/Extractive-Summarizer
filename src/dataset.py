import nltk
from typing import List, Tuple
from datasets import load_dataset
from rouge_score import rouge_scorer
from sentence_transformers import InputExample


def load_evaluation_dataset(lang: str = 'en', sample_count: int = 200):
    """
    Loads evaluation dataset:
    - English: CNN/DailyMail v3.0.0
    - Vietnamese: Fallback/VietNews sample
    """
    print(f"Loading {lang.upper()} evaluation dataset (samples={sample_count})...")
    if lang == 'en':
        ds = load_dataset("cnn_dailymail", "3.0.0", split="test")
        samples = ds.select(range(min(sample_count, len(ds))))
        return [
            {
                'id': idx,
                'article': sample['article'],
                'highlights': sample['highlights']
            }
            for idx, sample in enumerate(samples)
        ]
    else:
        # Fallback/Vietnamese dataset
        try:
            ds = load_dataset("vietnews", split="test")
            samples = ds.select(range(min(sample_count, len(ds))))
            return [
                {
                    'id': idx,
                    'article': sample['article'],
                    'highlights': sample['abstract']
                }
                for idx, sample in enumerate(samples)
            ]
        except Exception:
            print("VietNews dataset not directly available. Using sample data structure.")
            return []


def generate_oracle_extractive_pairs(articles_data: List[dict], max_pairs: int = 2000) -> List[InputExample]:
    """
    Generates Oracle Extractive Pairs (Article Sentence <-> Reference Summary Sentence)
    using ROUGE-1 scores as labels (Pull if >0.45, Push if <0.10).
    """
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    training_examples = []

    print(f"Generating Oracle Extractive Pairs for Fine-Tuning (target: {max_pairs})...")
    for sample in articles_data:
        if len(training_examples) >= max_pairs:
            break

        article_sents = nltk.sent_tokenize(sample['article'])
        summary_sents = nltk.sent_tokenize(sample['highlights'])

        for a_sent in article_sents[:8]:  # Focus on lead sentences
            for s_sent in summary_sents:
                score = float(scorer.score(s_sent, a_sent)['rouge1'].fmeasure)
                
                # Pull pair (label = 1.0) if strong overlap
                if score > 0.45:
                    training_examples.append(InputExample(texts=[a_sent, s_sent], label=1.0))
                # Push pair (label = 0.0) if negligible overlap
                elif score < 0.10:
                    training_examples.append(InputExample(texts=[a_sent, s_sent], label=0.0))

                if len(training_examples) >= max_pairs:
                    break

    print(f"Successfully generated {len(training_examples)} Oracle training pairs.")
    return training_examples
