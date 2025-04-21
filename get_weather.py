from os import write
import requests
from lxml import etree
from headers import *
from xpinyin import Pinyin
import csv
from lxml.html import fromstring, tostring
import re
from bs4 import BeautifulSoup

session = requests.Session()
week = 7

# 正则表达式模式 找正数和负数
pattern1 = r'[-+]?\d*\.?\d+'
# 正则表达式模式 找“，”和“。”中间的内容
pattern2 = r'(?<=，).+?(?=。)'
# 正则表达式模式 找冒号后面的内容
pattern3 = r":(.*)"
humidity_pattern = re.compile(r'湿度：(\d+%)')
wind_direction_pattern = re.compile(r'风向：(.+?)(?=级)')
wind_level_pattern = re.compile(r'(\d+)级')
Urays_pattern = re.compile(r'紫外线：(.+)')
air_quality_pattern = re.compile(r'空气质量：(.+)')

def get_city_url():
    """
    找到所查询的城市天气的网址。天气网
    """
    search_city = '北京'
    P = Pinyin()
    pinyin = P.get_pinyin(search_city, '')
    front_url = 'https://www.tianqi.com/'
    com_url = front_url + pinyin + '/7/'
    index = session.get(com_url, headers=headers)
    index.encoding = 'utf-8'
    html = etree.HTML(index.text)
    # print(target_city_url)
    urls = html.xpath("//body//div[@class='inleft']//ul//a[contains(@title, '北京')]//@href")
    new_urls = []
    for url in urls:
        new_urls.append(front_url + url)
    return new_urls[0:week]


def get_weather_info():
    time_rule = "//div[@class='weatherbox']//dl[@class='weather_info']//dd[@class='week']/text()"
    temperature_rule = "//div[@class='weatherbox']//dl[@class='weather_info']//dd[@class='weather']/span/text()"
    humidity_rule = "//div[@class='weatherbox']//dl[@class='weather_info']//dd[@class='shidu']//b//text()"
    air_rule = "//div[@class='weatherbox']//dl[@class='weather_info']//dd[@class='kongqi']//text()"
    csv_file = open("weather_data1.csv", "w", newline='', encoding='utf-8')
    writer = csv.writer(csv_file)
    writer.writerow(['时间', '最低温度/℃', '最高温度/℃', '湿度/%', '风向', '风力', '紫外线', '空气质量'])
    for each in get_city_url()[0: 3]:  # 前三个链接相同规则，后四个另一个规则
        content = session.get(each, headers=headers)
        content.encoding = 'utf-8'
        html = etree.HTML(content.text)
        time_res = str(html.xpath(time_rule)[0])[0:11]
        low_temperature = re.findall(pattern1, str(html.xpath(temperature_rule)[0]))[0]
        high_temperature = re.findall(pattern1, str(html.xpath(temperature_rule)[0]))[1]
        humidity = re.sub('\\D', '', (re.findall(humidity_pattern, html.xpath(humidity_rule)[0])[0]))
        wind_direction = str(re.findall(wind_direction_pattern, html.xpath(humidity_rule)[1])[0]).split(" ")[0]
        wind_level = re.sub('\\D', '', (re.findall(wind_level_pattern, html.xpath(humidity_rule)[1])[0]))
        Urays = re.findall(Urays_pattern, html.xpath(humidity_rule)[2])[0]
        air_quality = re.findall(air_quality_pattern, str(html.xpath(air_rule)[0]))[0]
        data = [time_res, low_temperature, high_temperature, humidity, wind_direction, wind_level, Urays, air_quality]
        writer.writerow(data)
        # print(time_res)
        # print(low_temperature)
        # print(high_temperature)
        # print(humidity)
        # print(wind_direction)
        # print(wind_level)
        # print(Urays)
        # print(air_quality)

    for each in get_city_url()[3: week]:
        new_time_rule = "//body//div[@class='mainbox clearfix']//div[@class='tips_proverb']/text()"
        new_other_rule = "//body//div[@class='mainbox clearfix']//div[@class='tips_pretext']/span"
        new_air_rule = "//body//div[@class='mainbox clearfix']//div[@class='tips_pretext']/text()[8]"
        content = session.get(each, headers=headers)
        content.encoding = 'utf-8'
        html = etree.HTML(content.text)
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
        data = [time_res, low_temperature, high_temperature, humidity, wind_direction, wind_level, Urays, air_quality]
        writer.writerow(data)


if __name__ == '__main__':
    get_weather_info()
