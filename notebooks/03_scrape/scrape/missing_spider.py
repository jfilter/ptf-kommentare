import scrapy
from bs4 import BeautifulSoup

class MissingComments(scrapy.Spider):
    name='more_comments'
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.135 Safari/537.36 Edge/12.246'
    start_urls = [l.strip() for l in open('missing.txt').readlines()]

    custom_settings = {'FEED_FORMAT': 'jsonlines', 'FEED_URI': '/root/missing_more.json'}

    def parse(self, response):
        doc = BeautifulSoup(response.text)
        comments = doc.find_all('article', {"class": "comment"})
        co = []
        for c in comments:
            body = c.find('div', {'class': 'comment__body'}).get_text().strip()
            date = c.find('a', {'class': 'comment-meta__date'}).get_text().strip()
            co.append({'text': body, 'date': date})

        for lm in doc.find_all('div', {'class': 'js-comment-loader'}):
            yield {'more_url': lm.find('a')['data-url'] }

        yield {'url': response.url, 'comments': co}
