from typing import Dict
import requests
from bs4 import BeautifulSoup

try:
    from newspaper import Article
except ImportError:
    Article = None


def crawl_article_from_url(url: str) -> Dict[str, str]:
    """
    Crawls news article title and text from URL using newspaper3k or BeautifulSoup.
    """
    if Article is not None:
        try:
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
            print(f"newspaper3k failed for {url}: {e}. Fallback to BeautifulSoup...")

    # Fallback to BeautifulSoup HTML parsing
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, 'html.parser')

        title = soup.find('h1')
        title_text = title.get_text().strip() if title else 'Bài báo trích xuất'

        paragraphs = soup.find_all('p')
        text_content = "\n".join([p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 30])

        if not text_content:
            text_content = soup.get_text()

        return {
            'title': title_text,
            'text': text_content,
            'url': url
        }
    except Exception as err:
        raise ValueError(f"Không thể cào dữ liệu từ đường link URL này: {err}")
