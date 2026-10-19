from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class TextSummarizeRequest(BaseModel):
    text: str = Field(..., description="Nội dung đoạn văn bản cần tóm tắt")
    lang: Literal['auto', 'vi', 'en'] = 'auto'
    length: Literal['brief', 'medium', 'detailed'] = 'medium'


class UrlSummarizeRequest(BaseModel):
    url: str = Field(..., description="Đường link bài báo (VNExpress, Dân Trí, CNN...)")
    lang: Literal['auto', 'vi', 'en'] = 'auto'
    length: Literal['brief', 'medium', 'detailed'] = 'medium'


class TopicSummarizeRequest(BaseModel):
    topic: str = Field(..., description="Từ khóa chủ đề cần tóm tắt nhiều bài báo")
    lang: Literal['vi', 'en'] = 'vi'
    length: Literal['brief', 'medium', 'detailed'] = 'medium'


class StatsSchema(BaseModel):
    original_sentence_count: int
    summary_sentence_count: int
    compression_ratio: float  # Tỷ lệ nén %
    latency_ms: float         # Thời gian xử lý ms
    silhouette_score: Optional[float] = None
    diversity_score: Optional[float] = None
    detected_language: str    # Ngôn ngữ nhận diện được
    method: str
    model_version: str
    summary_policy: str
    note: str


class SentenceSpan(BaseModel):
    index: int
    start: int
    end: int


class SummarizeResponse(BaseModel):
    title: Optional[str] = None
    source_url: Optional[str] = None
    original_text: str
    original_sentence_spans: List[SentenceSpan] = Field(default_factory=list)
    summary_text: str
    summary_sentences: List[str]
    highlight_indices: List[int]
    stats: StatsSchema
    is_today_news: Optional[bool] = True
    date_range: Optional[str] = None
    notice: Optional[str] = None
