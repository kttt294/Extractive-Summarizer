from typing import Dict, Any
import ssl
import urllib3
import requests
from bs4 import BeautifulSoup

# Disable SSL warnings for legacy news sites
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class LegacySSLAdapter(requests.adapters.HTTPAdapter):
    """
    Custom HTTPAdapter enabling OP_LEGACY_SERVER_CONNECT to allow scraping
    older SSL/TLS news sites (such as baotintuc.vn) in modern Python/OpenSSL environments.
    """
    def init_poolmanager(self, *args, **kwargs):
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        if hasattr(ssl, 'OP_LEGACY_SERVER_CONNECT'):
            ctx.options |= ssl.OP_LEGACY_SERVER_CONNECT
        else:
            ctx.options |= 0x4
        kwargs['ssl_context'] = ctx
        return super().init_poolmanager(*args, **kwargs)

def get_robust_session() -> requests.Session:
    session = requests.Session()
    adapter = LegacySSLAdapter()
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    return session

try:
    from newspaper import Article
except ImportError:
    Article = None


def crawl_article_from_url(url: str) -> Dict[str, str]:
    """
    Crawls news article title and text from URL using dedicated Vietnamese news parser + newspaper3k fallback.
    """
    session = get_robust_session()
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}

    html_content = None
    try:
        resp = session.get(url, headers=headers, timeout=12, verify=False)
        resp.encoding = resp.apparent_encoding or 'utf-8'
        html_content = resp.text
    except Exception as fetch_err:
        print(f"Direct session fetch failed for {url}: {fetch_err}")

    # 1. Primary: Dedicated BeautifulSoup extraction targeting Sapo + Content Body + Caption Filtering
    if html_content:
        try:
            soup = BeautifulSoup(html_content, 'html.parser')

            # Remove noise tags
            for elem in soup(["script", "style", "nav", "footer", "header", "iframe", "form", "svg"]):
                elem.decompose()

            title = soup.find('h1')
            title_text = title.get_text().strip() if title else 'Bài báo trích xuất'

            extracted_lines = []

            # 1.1 Sapo / Lead Paragraph (common in VN news: .sapo, .detail-sapo, .lead, h2.sapo, .bold)
            sapo_tag = soup.select_one('.sapo, .detail-sapo, .lead, h2.sapo, div.sapo, p.sapo, div.bold, p.bold')
            if sapo_tag:
                sapo_text = sapo_tag.get_text().strip()
                if len(sapo_text) > 20 and not any(k in sapo_text for k in ['Ảnh minh họa:', 'Ảnh:', 'Nguồn:']):
                    extracted_lines.append(sapo_text)

            # 1.2 Main Content Body Container (.contents, .detail-content, .fck_detail, article, main)
            content_container = soup.select_one('.contents, .detail-content, .fck_detail, .detail__content, .content, article, main')
            target_root = content_container if content_container else soup

            for elem in target_root.find_all(['p', 'h2', 'h3']):
                txt = elem.get_text().strip()

                # Ignore image captions, photo credits, and source metadata
                if any(k in txt for k in ['Ảnh minh họa:', 'Ảnh:', 'Nguồn:', 'TTXVN', 'Ảnh/Clip:']):
                    if len(txt) < 100:
                        continue

                # Ignore short noise or duplicated sapo
                if len(txt) > 30 and txt not in extracted_lines:
                    extracted_lines.append(txt)

            text_content = "\n\n".join(extracted_lines)

            if text_content and len(text_content.strip()) > 100:
                return {
                    'title': title_text,
                    'text': text_content,
                    'url': url
                }
        except Exception as bs_err:
            print(f"BeautifulSoup parsing failed: {bs_err}. Trying newspaper3k...")

    # 2. Fallback to newspaper3k parsing
    if Article is not None:
        try:
            if html_content:
                article = Article(url)
                article.set_html(html_content)
                article.parse()
            else:
                article = Article(url)
                article.download()
                article.parse()

            if article.text and len(article.text.strip()) > 100:
                return {
                    'title': article.title or 'Bài báo trích xuất',
                    'text': article.text,
                    'url': url
                }
        except Exception as e:
            print(f"newspaper3k failed for {url}: {e}")

    raise ValueError("Không thể cào dữ liệu từ đường link URL này hoặc nội dung bài viết quá ngắn.")


def crawl_articles_from_topic(topic: str, lang: str = 'vi') -> Dict[str, Any]:
    """
    Cào và tổng hợp bài viết mới nhất theo từ khóa chủ đề (DuckDuckGo News API + Fallback HTML Scraping).
    """
    combined_texts = []
    dates_found = []
    is_today_news = True
    notice_msg = None

    region_map = {
        'vi': 'vn-vi',
        'en': 'us-en',
        'auto': 'wt-wt'
    }
    ddg_region = region_map.get(lang, 'wt-wt')

    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            # 1. Thử lấy tin tức mới nhất trong ngày (timelimit='d') với vùng ngôn ngữ tương ứng
            results = list(ddgs.news(topic, region=ddg_region, timelimit='d', max_results=5))

            # 2. Nếu không đủ bài trong 24h, tự động mở rộng trong tuần (timelimit='w')
            if len(results) < 2:
                is_today_news = False
                notice_msg = "Không tìm thấy bài viết mới trong ngày hôm nay. Hệ thống đã tự động tổng hợp từ các bài viết mới nhất trong tuần qua."
                results = list(ddgs.news(topic, region=ddg_region, timelimit='w', max_results=5))

                if not results:
                    results = list(ddgs.text(topic, region=ddg_region, max_results=5))

            for item in results:
                text_body = item.get('body') or item.get('snippet')
                if text_body and len(text_body) > 30:
                    combined_texts.append(text_body)
                if 'date' in item and item['date']:
                    dates_found.append(item['date'])
    except Exception as ddg_err:
        print(f"DDGS API error: {ddg_err}. Falling back to HTML scraping...")

    # Fallback HTML scraping nếu API không trả về
    if not combined_texts:
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            search_url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(topic)}"
            resp = requests.get(search_url, headers=headers, timeout=10)
            soup = BeautifulSoup(resp.text, 'html.parser')
            snippets = soup.find_all('a', class_='result__snippet')
            for snip in snippets[:5]:
                text = snip.get_text().strip()
                if len(text) > 30:
                    combined_texts.append(text)
            is_today_news = False
        except Exception as err:
            raise ValueError(f"Không thể tìm thấy tin tức về chủ đề '{topic}': {err}")

    if not combined_texts:
        raise ValueError(f"Không tìm thấy bài viết hoặc tin tức nào phù hợp với chủ đề '{topic}'.")

    full_text = "\n\n".join(combined_texts)
    date_range_str = None
    if dates_found:
        from datetime import datetime, timezone, timedelta
        vn_tz = timezone(timedelta(hours=7))
        parsed_dates = []
        for d in dates_found:
            try:
                dt = datetime.fromisoformat(d.replace('Z', '+00:00')).astimezone(vn_tz)
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
            date_range_str = f"Hôm nay ({datetime.now(vn_tz).strftime('%d/%m/%Y')})"
    else:
        from datetime import datetime, timezone, timedelta
        vn_tz = timezone(timedelta(hours=7))
        date_range_str = f"Hôm nay ({datetime.now(vn_tz).strftime('%d/%m/%Y')})"

    return {
        'topic': topic,
        'combined_text': full_text,
        'is_today_news': is_today_news,
        'date_range': date_range_str,
        'notice': notice_msg
    }
