from fastapi import APIRouter, HTTPException
from backend.schemas.payload import TextSummarizeRequest, UrlSummarizeRequest, TopicSummarizeRequest, SummarizeResponse
from backend.services.crawler import crawl_article_from_url, crawl_articles_from_topic
from backend.services.summarizer_service import process_summarization
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

router = APIRouter(prefix="/api/v1/summarize", tags=["Summarize"])


@router.post("/text", response_model=SummarizeResponse)
def summarize_text(payload: TextSummarizeRequest):
    """
    Tóm tắt đoạn văn bản thô trực tiếp
    """
    try:
        res = process_summarization(
            text=payload.text,
            user_lang=payload.lang,
            length=payload.length
        )
        return SummarizeResponse(**res)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi hệ thống khi tóm tắt: {e}")


@router.post("/url", response_model=SummarizeResponse)
def summarize_url(payload: UrlSummarizeRequest):
    """
    Crawl và Tóm tắt bài báo từ đường link URL
    """
    try:
        crawled = crawl_article_from_url(payload.url)
        res = process_summarization(
            text=crawled['text'],
            user_lang=payload.lang,
            length=payload.length
        )
        res['title'] = crawled['title']
        res['source_url'] = payload.url
        return SummarizeResponse(**res)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi crawl hoặc tóm tắt đường link URL: {e}")


@router.post("/topic", response_model=SummarizeResponse)
def summarize_topic(payload: TopicSummarizeRequest):
    """
    Cào bài viết và Tóm tắt đa bài viết theo từ khóa chủ đề.
    """
    try:
        crawled = crawl_articles_from_topic(payload.topic, lang=payload.lang)
        res = process_summarization(
            text=crawled['combined_text'],
            user_lang=payload.lang,
            length=payload.length
        )
        res['title'] = f"Tổng hợp chủ đề: {payload.topic}"
        res['is_today_news'] = crawled['is_today_news']
        res['date_range'] = crawled['date_range']
        res['notice'] = crawled['notice']
        return SummarizeResponse(**res)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi cào dữ liệu hoặc tóm tắt chủ đề: {e}")
