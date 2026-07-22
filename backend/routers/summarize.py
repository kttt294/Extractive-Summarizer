from fastapi import APIRouter, HTTPException
from backend.schemas.payload import TextSummarizeRequest, UrlSummarizeRequest, TopicSummarizeRequest, SummarizeResponse
from backend.services.crawler import crawl_article_from_url
from backend.services.summarizer_service import process_summarization
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

router = APIRouter(prefix="/api/v1/summarize", tags=["Summarize"])


@router.post("/text", response_model=SummarizeResponse)
def summarize_text(payload: TextSummarizeRequest):
    """
    Tóm tắt đoạn văn bản thô trực tiếp.
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
    Crawl và Tóm tắt bài báo từ đường link URL.
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
    Tóm tắt đa bài viết theo từ khóa chủ đề (Ưu tiên bài trong ngày -> Mở rộng bài cũ hơn + Thông báo + Thanh timeline).
    """
    try:
        combined_texts = []
        dates_found = []
        is_today_news = True
        notice_msg = None
        
        # Thử lấy tin tức trong ngày (timelimit='d')
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.news(payload.topic, timelimit='d', max_results=5))
                
                # Nếu không tìm thấy hoặc ít bài trong 24h qua, mở rộng lấy trong tuần/tháng (timelimit='w')
                if len(results) < 2:
                    is_today_news = False
                    notice_msg = "Không tìm thấy bài viết mới trong ngày hôm nay. Hệ thống đã tự động tổng hợp từ các bài viết mới nhất trong tuần qua."
                    results = list(ddgs.news(payload.topic, timelimit='w', max_results=5))
                    
                    if not results:
                        results = list(ddgs.text(payload.topic, max_results=5))

                for item in results:
                    text_body = item.get('body') or item.get('snippet')
                    if text_body and len(text_body) > 30:
                        combined_texts.append(text_body)
                    
                    # Thu thập ngày xuất bản nếu có
                    if 'date' in item and item['date']:
                        dates_found.append(item['date'])
        except Exception as ddg_err:
            print(f"DDGS API error: {ddg_err}. Falling back to HTML scraping...")

        # Fallback HTML scraping nếu API không trả về
        if not combined_texts:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            search_url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(payload.topic)}"
            resp = requests.get(search_url, headers=headers, timeout=10)
            soup = BeautifulSoup(resp.text, 'html.parser')
            snippets = soup.find_all('a', class_='result__snippet')
            for snip in snippets[:5]:
                text = snip.get_text().strip()
                if len(text) > 30:
                    combined_texts.append(text)
            
            is_today_news = False
            notice_msg = "Tổng hợp dữ liệu bài viết mới nhất từ kết quả tìm kiếm web."

        if not combined_texts:
            raise ValueError(f"Không thể lấy được bài viết nào cho chủ đề: '{payload.topic}'. Vui lòng thử từ khóa khác.")

        # Xử lý dải thời gian từ bài cũ nhất đến bài mới nhất
        today_str = datetime.now().strftime("%d/%m/%Y")
        if dates_found:
            parsed_dates = []
            for d in dates_found:
                try:
                    # ISO string format '2026-07-22T12:00:00+00:00'
                    dt = datetime.fromisoformat(d.replace('Z', '+00:00'))
                    parsed_dates.append(dt)
                except Exception:
                    pass
            
            if parsed_dates:
                min_date = min(parsed_dates).strftime("%d/%m/%Y")
                max_date = max(parsed_dates).strftime("%d/%m/%Y")
                if min_date == max_date:
                    date_range_str = f"Ngày {min_date}"
                else:
                    date_range_str = f"{min_date} – {max_date}"
            else:
                date_range_str = f"Trong tuần đến {today_str}"
        else:
            if is_today_news:
                date_range_str = f"Hôm nay ({today_str})"
            else:
                date_range_str = f"3-7 ngày qua đến {today_str}"

        text_content = "\n\n".join(combined_texts)
        res = process_summarization(
            text=text_content,
            user_lang=payload.lang,
            length=payload.length
        )
        
        res['title'] = f"Tóm tắt chủ đề: '{payload.topic}'"
        res['is_today_news'] = is_today_news
        res['date_range'] = date_range_str
        res['notice'] = notice_msg
        
        return SummarizeResponse(**res)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi tìm kiếm tóm tắt chủ đề: {e}")
