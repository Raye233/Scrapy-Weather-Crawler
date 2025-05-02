# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy

class day7_WeatherItem(scrapy.Item):
    时间 = scrapy.Field()
    最低温度 = scrapy.Field()
    最高温度 = scrapy.Field()
    湿度 = scrapy.Field()
    风向 = scrapy.Field()
    风力 = scrapy.Field()
    紫外线 = scrapy.Field()
    空气质量 = scrapy.Field()

class year3_WeatherItem(scrapy.Item):
    时间 = scrapy.Field()
    最低温度 = scrapy.Field()
    最高温度 = scrapy.Field()
    湿度 = scrapy.Field()
    风向 = scrapy.Field()
    风力 = scrapy.Field()
    紫外线 = scrapy.Field()
    空气质量 = scrapy.Field()

class year5_WeatherItem(scrapy.Item):
    时间 = scrapy.Field()
    最低温度 = scrapy.Field()
    最高温度 = scrapy.Field()
    湿度 = scrapy.Field()
    风向 = scrapy.Field()
    风力 = scrapy.Field()
    紫外线 = scrapy.Field()
    空气质量 = scrapy.Field()
