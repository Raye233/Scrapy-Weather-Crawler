# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html
import codecs
# useful for handling different item types with a single interface
import csv
import os
import subprocess

class day7_WeatherPipeline:
    def open_spider(self, spider):
        self.csv_file = open("day7_weather_data.csv", "w", newline='', encoding='utf-8')
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(['时间', '最低温度/℃', '最高温度/℃', '湿度/%', '风向', '风力', '紫外线', '空气质量'])

    def close_spider(self, spider):
        self.csv_file.close()
        script_path = os.path.abspath('sort.py')

        try:
            # 执行 sort.py
            subprocess.run(['python', script_path], check=True)
            print("sort.py executed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error executing sort.py: {e}")

    def process_item(self, item, spider):
        self.writer.writerow([
            item['时间'],
            item['最低温度'],
            item['最高温度'],
            item['湿度'],
            item['风向'],
            item['风力'],
            item['紫外线'],
            item['空气质量']
        ])
        return item


class year3_WeatherPipeline:
    def open_spider(self, spider):
        self.csv_file = open("year3_weather_data.csv", "a+", newline='')
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(['时间', '最低温度/℃', '最高温度/℃', '湿度/%', '风向', '风力', '紫外线', '空气质量'])

    def close_spider(self, spider):
        self.csv_file.close()
        script_path = os.path.abspath('sort.py')

        try:
            # 执行 sort.py
            subprocess.run(['python', script_path], check=True)
            print("sort.py executed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error executing sort.py: {e}")

    def process_item(self, item, spider):
        self.writer.writerow([
            item['时间'],
            item['最低温度'],
            item['最高温度'],
            item['湿度'],
            item['风向'],
            item['风力'],
            item['紫外线'],
            item['空气质量']
        ])
        return item

