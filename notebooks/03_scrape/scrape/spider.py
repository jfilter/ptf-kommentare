from bs4 import BeautifulSoup

import scrapy
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor

IGNORED_EXTENSIONS = [
    # images
    'mng', 'pct', 'bmp', 'gif', 'jpg', 'jpeg', 'png', 'pst', 'psp', 'tif',
    'tiff', 'ai', 'drw', 'dxf', 'eps', 'ps', 'svg',

    # audio
    'mp3', 'wma', 'ogg', 'wav', 'ra', 'aac', 'mid', 'au', 'aiff',

    # video
    '3gp', 'asf', 'asx', 'avi', 'mov', 'mp4', 'mpg', 'qt', 'rm', 'swf', 'wmv',
    'm4a', 'm4v', 'flv',

    # office suites
    'xls', 'xlsx', 'ppt', 'pptx', 'pps', 'doc', 'docx', 'odt', 'ods', 'odg',
    'odp',

    # other
    'css', 'pdf', 'exe', 'bin', 'rss', 'zip', 'rar',
]


class MySpider(CrawlSpider):
    name = 'zeit.de'
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.135 Safari/537.36 Edge/12.246'
    allowed_domains = ['zeit.de']
    start_urls = ['https://www.zeit.de']

    rules = (
        Rule(LinkExtractor(allow=r'(^[^\?]+$)|(^.*\?page.*$)', deny_extensions=IGNORED_EXTENSIONS), callback='parse_item', follow=True),
    )

    custom_settings = {
        'LOG_LEVEL':'INFO',
        'BOT_NAME': 'MOZILLA',
        'FEED_FORMAT': 'jsonlines', 'FEED_URI': '/root/data_run2.json'}

    def parse_item(self, response):
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
