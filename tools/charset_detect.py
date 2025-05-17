import chardet

# filepath = r'F:\Rayedata2\weather_crawl\wuhan_year5_weather_data.csv'

def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        result = chardet.detect(file.read())
        # if result['encoding'] == 'GB2312':
        #     result['encoding'] = 'gbk'
        print("训练集详细编码格式为:", result['encoding'])
        #     return result['encoding']
    return result['encoding']


# detect_encoding(filepath)
