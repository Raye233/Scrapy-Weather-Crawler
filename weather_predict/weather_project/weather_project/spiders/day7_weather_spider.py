from re import search

import scrapy
import re
from lxml import etree
from scrapy import Spider
from twisted.internet.defer import Deferred
import os
from ..items import day7_WeatherItem
from xpinyin import Pinyin
import subprocess
from datetime import datetime, timedelta

# 正则表达式模式
pattern1 = r'[-+]?\d*\.?\d+'
pattern2 = r'(?<=，).+?(?=。)'
pattern3 = r":(.*)"
humidity_pattern = re.compile(r'湿度：(\d+%)')
wind_direction_pattern = re.compile(r'风向：(.+?)(?=级)')
wind_level_pattern = re.compile(r'(\d+)级')
Urays_pattern = re.compile(r'紫外线：(.+)')
air_quality_pattern = re.compile(r'空气质量：(.+)')


class day7_WeatherSpider(scrapy.Spider):
    name = 'day7_weather'
    allowed_domains = ['tianqi.com']
    start_urls = ['https://www.tianqi.com/']

    def __init__(self, *args, **kwargs):
        super(day7_WeatherSpider, self).__init__(*args, **kwargs)
        self.p = Pinyin()
        self.week = 7
        self.interrupt = 3
        self.city_name = kwargs.get('city_name')

    def start_requests(self):
        print("== 爬虫开始请求 ==")
        search_city = self.city_name
        pinyin = self.p.get_pinyin(search_city, '')
        front_url = 'https://www.tianqi.com/'
        com_url = f"{front_url}{pinyin}/7/"
        # print(com_url)  #  正常
        yield scrapy.Request(url=com_url, callback=self.parse_city_url)

    def parse_city_url(self, response):
        response = response.replace(encoding='utf-8')
        # print(response.text)  # 正常
        html_string = etree.HTML(response.text)
        # urls = response.xpath("//body//div[@class='inleft']//ul//a[contains(@title, '北京')]//@href").getall()
        urls = html_string.xpath(f"//body//div[@class='inleft']//ul//a[contains(@title, '{self.city_name}')]//@href")
        new_urls = [response.urljoin(url) for url in urls]
        # print(new_urls)
        for url in new_urls[:self.interrupt]:
            yield scrapy.Request(url=url, callback=self.three_parse_weather_info)
        for url in new_urls[self.interrupt: self.week]:
            yield scrapy.Request(url=url, callback=self.four_parse_weather_info)

    def three_parse_weather_info(self, response):
        item = day7_WeatherItem()
        time_rule = "//div[@class='weatherbox']//dl[@class='weather_info']//dd[@class='week']/text()"
        temperature_rule = "//div[@class='weatherbox']//dl[@class='weather_info']//dd[@class='weather']/span/text()"
        humidity_rule = "//div[@class='weatherbox']//dl[@class='weather_info']//dd[@class='shidu']//b//text()"
        air_rule = "//div[@class='weatherbox']//dl[@class='weather_info']//dd[@class='kongqi']//text()"

        response = response.replace(encoding='utf-8')
        html_string = etree.HTML(response.text)
        time_res = str(html_string.xpath(time_rule)[0])[0:11]
        low_temperature = re.findall(pattern1, str(html_string.xpath(temperature_rule)[0]))[0]
        high_temperature = re.findall(pattern1, str(html_string.xpath(temperature_rule)[0]))[1]
        humidity = re.sub('\\D', '', (re.findall(humidity_pattern, html_string.xpath(humidity_rule)[0])[0]))
        wind_direction = str(re.findall(wind_direction_pattern, html_string.xpath(humidity_rule)[1])[0]).split(" ")[0]
        wind_level = re.sub('\\D', '', (re.findall(wind_level_pattern, html_string.xpath(humidity_rule)[1])[0]))
        Urays = re.findall(Urays_pattern, html_string.xpath(humidity_rule)[2])[0]
        air_quality = re.findall(air_quality_pattern, str(html_string.xpath(air_rule)[0]))[0]

        item['时间'] = time_res
        item['最低温度'] = low_temperature
        item['最高温度'] = high_temperature
        item['湿度'] = humidity
        item['风向'] = wind_direction
        item['风力'] = wind_level
        item['紫外线'] = Urays
        item['空气质量'] = air_quality

        yield item

    def four_parse_weather_info(self, response):
        item = day7_WeatherItem()
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


