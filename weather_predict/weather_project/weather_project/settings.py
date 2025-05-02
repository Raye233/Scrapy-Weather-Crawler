# Scrapy settings for weather_project project
#
# For simplicity, this file contains only settings considered important or
# commonly used. You can find more settings consulting the documentation:
#
#     https://docs.scrapy.org/en/latest/topics/settings.html
#     https://docs.scrapy.org/en/latest/topics/downloader-middleware.html
#     https://docs.scrapy.org/en/latest/topics/spider-middleware.html

BOT_NAME = "weather_project"

SPIDER_MODULES = ["weather_project.spiders"]
NEWSPIDER_MODULE = "weather_project.spiders"

# Crawl responsibly by identifying yourself (and your website) on the user-agent
#USER_AGENT = "weather_project (+http://www.yourdomain.com)"

# Obey robots.txt rules
# 启用调试日志并输出到文件
LOG_LEVEL = 'DEBUG'
LOG_FILE = 'scrapy_debug.log'
DOWNLOADER_DEBUG = True  # 显示请求头等详细信息
ROBOTSTXT_OBEY = False   # 确认关闭robots.txt检查

# Configure maximum concurrent requests performed by Scrapy (default: 16)
CONCURRENT_REQUESTS = 4

# Configure a delay for requests for the same website (default: 0)
# See https://docs.scrapy.org/en/latest/topics/settings.html#download-delay
# See also autothrottle settings and docs
DOWNLOAD_DELAY = 2
# The download delay setting will honor only one of:
#CONCURRENT_REQUESTS_PER_DOMAIN = 16
#CONCURRENT_REQUESTS_PER_IP = 16

# Disable cookies (enabled by default)
#COOKIES_ENABLED = False

# Disable Telnet Console (enabled by default)
#TELNETCONSOLE_ENABLED = False

# Override the default request headers:
DEFAULT_REQUEST_HEADERS = {
    'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0",
    'Accept': "application/json, text/javascript, */*; q=0.01",
    'accept-language': "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
    'accept-encoding': "gzip, deflate, br, zstd",
    'referer': "https://www.tianqi.com/",
    'cookie': "UserId=17356387988638478; cityPy=jingzhou; cityPy_expire=1736243599; Hm_lvt_ab6a683aa97a52202eab5b3a9042a8d2=1735638799; HMACCOUNT=16B5032E4A72B95E; Hm_lvt_30606b57e40fddacb2c26d2b789efbcb=1735638848; Hm_lpvt_30606b57e40fddacb2c26d2b789efbcb=1735639408; Hm_lpvt_ab6a683aa97a52202eab5b3a9042a8d2=1735639487",
}
# Enable or disable spider middlewares
# See https://docs.scrapy.org/en/latest/topics/spider-middleware.html
# SPIDER_MIDDLEWARES = {
#    "weather_project.middlewares.WeatherProjectSpiderMiddleware": 543,
# }

# Enable or disable downloader middlewares
# See https://docs.scrapy.org/en/latest/topics/downloader-middleware.html
#DOWNLOADER_MIDDLEWARES = {
#    "weather_project.middlewares.WeatherProjectDownloaderMiddleware": 543,
#}

# Enable or disable extensions
# See https://docs.scrapy.org/en/latest/topics/extensions.html
#EXTENSIONS = {
#    "scrapy.extensions.telnet.TelnetConsole": None,
#}

# Configure item pipelines
# See https://docs.scrapy.org/en/latest/topics/item-pipeline.html
ITEM_PIPELINES = {
    'weather_project.pipelines.day7_WeatherPipeline': 300,
    'weather_project.pipelines.year3_WeatherPipeline': 400,
    'weather_project.pipelines.year5_WeatherPipeline': 500,
}

# Enable and configure the AutoThrottle extension (disabled by default)
# See https://docs.scrapy.org/en/latest/topics/autothrottle.html
#AUTOTHROTTLE_ENABLED = True
# The initial download delay
#AUTOTHROTTLE_START_DELAY = 5
# The maximum download delay to be set in case of high latencies
#AUTOTHROTTLE_MAX_DELAY = 60
# The average number of requests Scrapy should be sending in parallel to
# each remote server
#AUTOTHROTTLE_TARGET_CONCURRENCY = 1.0
# Enable showing throttling stats for every response received:
#AUTOTHROTTLE_DEBUG = False

# Enable and configure HTTP caching (disabled by default)
# See https://docs.scrapy.org/en/latest/topics/downloader-middleware.html#httpcache-middleware-settings
#HTTPCACHE_ENABLED = True
#HTTPCACHE_EXPIRATION_SECS = 0
#HTTPCACHE_DIR = "httpcache"
#HTTPCACHE_IGNORE_HTTP_CODES = []
#HTTPCACHE_STORAGE = "scrapy.extensions.httpcache.FilesystemCacheStorage"

# Set settings whose default value is deprecated to a future-proof value
# REQUEST_FINGERPRINTER_IMPLEMENTATION = "2.7"
REQUEST_FINGERPRINTER_CLASS = "scrapy.utils.request.RequestFingerprinter"
TWISTED_REACTOR = "twisted.internet.asyncioreactor.AsyncioSelectorReactor"


FEED_EXPORT_ENCODING = "utf-8"

# settings.py
import random
from .proxy import proxy_pool


def get_random_proxy():
    return random.choice(proxy_pool)


# # 设置代理
# HTTP_PROXY = random.choice(proxy_pool)  # 替换为你的代理IP

# 启用中间件
DOWNLOADER_MIDDLEWARES = {
    'scrapy.downloadermiddlewares.httpproxy.HttpProxyMiddleware': 110,
    'scrapy.downloadermiddlewares.useragent.UserAgentMiddleware': None,
}
