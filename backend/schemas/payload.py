from typing import List, Optional
from pydantic import BaseModel, Field


class TextSummarizeRequest(BaseModel):
    text: str = Field(..., description="Nội dung đoạn văn bản cần tóm tắt")
    lang: str = Field("auto", description="Ngôn ngữ ('auto', 'vi', 'en')")
    length: str = Field("medium", description="Độ dài tóm tắt ('brief', 'medium', 'detailed')")


class UrlSummarizeRequest(BaseModel):
    url: str = Field(..., description="Đường link bài báo (VNExpress, Dân Trí, CNN...)")
    lang: str = Field("auto", description="Ngôn ngữ ('auto', 'vi', 'en')")
    length: str = Field("medium", description="Độ dài tóm tắt ('brief', 'medium', 'detailed')")


class TopicSummarizeRequest(BaseModel):
    topic: str = Field(..., description="Từ khóa chủ đề cần tóm tắt nhiều bài báo")
    lang: str = Field("vi", description="Ngôn ngữ ('vi', 'en')")
    length: str = Field("medium", description="Độ dài tóm tắt ('brief', 'medium', 'detailed')")


class StatsSchema(BaseModel):
    original_sentence_count: int
    summary_sentence_count: int
    compression_ratio: float  # Tỷ lệ nén %
    latency_ms: float         # Thời gian xử lý ms
    silhouette_score: float   # Chỉ số Nội tại Silhouette
    diversity_score: float    # Chỉ số Nội tại Diversity
    detected_language: str    # Ngôn ngữ nhận diện được


class SummarizeResponse(BaseModel):
    title: Optional[str] = None
    source_url: Optional[str] = None
    original_text: str
    summary_text: str
    summary_sentences: List[str]
    highlight_indices: List[int]
    stats: StatsSchema
    is_today_news: Optional[bool] = True
    date_range: Optional[str] = None
    notice: Optional[str] = None
