import scrapy
from bs4 import BeautifulSoup

class MoreComments(scrapy.Spider):
    name='more_comments'
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.135 Safari/537.36 Edge/12.246'
    start_urls = [l.strip() for l in open('urls.csv').readlines()]

    custom_settings = {'FEED_FORMAT': 'jsonlines', 'FEED_URI': '/root/more.json', 'LOG_LEVEL': 'INFO'}

    def parse(self, response):
        doc = BeautifulSoup(response.text)
        comments = doc.find_all('article', {"class": "comment"})
        res = []
        for c in comments:
            body = c.find('div', {'class': 'comment__body'}).get_text().strip()
            date = c.find('a', {'class': 'comment-meta__date'}).get_text().strip()
            res.append( {'text': body, 'date': date} )
        yield {'url': response.url, 'comments': res}
