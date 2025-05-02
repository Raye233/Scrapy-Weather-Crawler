import requests




proxies = {
    'http': 'http://122.10.82.237:80',
    'https': 'http://159.75.163.60:8080',
}

try:
    response = requests.get('https://www.baidu.com', proxies=proxies, timeout=10)
    response.raise_for_status()
    print("请求成功！")
    print(response.text)
except requests.exceptions.RequestException as e:
    print(f"请求失败：{e}")