import chardet
import pandas as pd

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']

file_path = r'F:\Rayedata2\weather_crawl\combined_file.csv'
encoding = detect_encoding(file_path)
df = pd.read_csv(file_path, encoding=encoding)
print(df)
print(detect_encoding(file_path))