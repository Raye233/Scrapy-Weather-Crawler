import scrapy
import re
from lxml import etree
from scrapy import Spider
from twisted.internet.defer import Deferred
import os
from ..items import recent5_WeatherItem
from xpinyin import Pinyin
import subprocess
from datetime import date, timedelta, datetime
import time

pattern1 = r'[-+]?\d*\.?\d+'
pattern2 = r'(?<=，).+?(?=。)'
pattern3 = r":(.*)"
humidity_pattern = re.compile(r'湿度：(\d+%)')
wind_direction_pattern = re.compile(r'风向：(.+?)(?=级)')
wind_level_pattern = re.compile(r'(\d+)级')
Urays_pattern = re.compile(r'紫外线：(.+)')
air_quality_pattern = re.compile(r'空气质量：(.+)')


class recent5_WeatherSpider(scrapy.Spider):
    name = 'recent5_weather'
    custom_settings = {
        'ITEM_PIPELINES': {
            'weather_project.pipelines.recent5_WeatherPipeline': 400,
        }
    }
    allowed_domains = ['tianqi.com']
    start_urls = ['https://www.tianqi.com/']

    def __init__(self, *args, **kwargs):
        super(recent5_WeatherSpider, self).__init__(*args, **kwargs)
        self.p = Pinyin()
        self.city_name = kwargs.get('city_name')

    def start_requests(self):
        print("== 爬虫开始请求 ==")
        search_city = self.city_name
        pinyin = self.p.get_pinyin(search_city, '')
        front_url = 'https://www.tianqi.com/'
        end_date = date.today()
        offset = timedelta(days=-4)
        start_date = end_date + offset
        current_date = start_date
        num = 0
        while current_date <= end_date:
            date_str = current_date.strftime('%Y%m%d')
            com_url = f"{front_url}/tianqi/{pinyin}/{date_str}/"
            print(com_url)  # 打印生成的 URL
            num += 1
            print("已抓取{}天数据".format(num))
            yield scrapy.Request(url=com_url, callback=self.five_days_parse_weather_info)
            current_date += timedelta(days=1)

    def five_days_parse_weather_info(self, response):
        item = recent5_WeatherItem()

        new_time_rule = "//body//div[@class='mainbox clearfix']//div[@class='tips_proverb']/text()"
        new_other_rule = "//body//div[@class='mainbox clearfix']//div[@class='tips_pretext']/span"
        new_air_rule = "//body//div[@class='mainbox clearfix']//div[@class='tips_pretext']/text()[8]"

        response = response.replace(encoding='utf-8')
        html = etree.HTML(response.text)
        time_res = str(html.xpath(new_time_rule)[0])[3: 14]
        other_res = html.xpath(new_other_rule)
        air_res = html.xpath(new_air_rule)[0]
        low_temperature = re.findall(pattern1, str(other_res[3].text))[0]
        high_temperature = re.findall(pattern1, str(other_res[2].text))[0]
        wind_direction = str(other_res[4].text).split(" ")[0]
        wind_level = str(other_res[4].text.split(" ")[1])[0]
        humidity = re.sub('\\D', '', str(other_res[5].text))
        Urays = str(other_res[6].text)
        air_quality = re.sub('\\D', '', re.search(pattern2, air_res).group())

        item['时间'] = time_res
        item['最低温度'] = low_temperature
        item['最高温度'] = high_temperature
        item['湿度'] = humidity
        item['风向'] = wind_direction
        item['风力'] = wind_level
        item['紫外线'] = Urays
        item['空气质量'] = air_quality

        yield item
