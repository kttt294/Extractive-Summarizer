"""ROUGE: giữ nguyên dấu tiếng Việt; tiếng Anh dùng Porter stemming."""
import re
import unicodedata

from rouge_score import rouge_scorer


class VietnameseTokenizer:
    """Token theo âm tiết Unicode, không tuyên bố là ROUGE theo từ ghép."""

    def tokenize(self, text):
        text = unicodedata.normalize('NFC', text).lower()
        return re.findall(r'[^\W_]+', text, flags=re.UNICODE)


def make_rouge_scorer(lang='vi', metrics=('rouge1', 'rouge2', 'rougeL')):
    if lang == 'vi':
        return rouge_scorer.RougeScorer(list(metrics), tokenizer=VietnameseTokenizer())
    if lang == 'en':
        return rouge_scorer.RougeScorer(list(metrics), use_stemmer=True)
    raise ValueError(f'Ngôn ngữ chưa hỗ trợ: {lang}')
