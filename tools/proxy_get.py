import requests

url = 'https://proxy.scdn.io/api/get_proxy.php'
params = {
    'protocol': 'http',
    'count': 5
}
response = requests.get(url, params=params)
data = response.json()
# with open('proxy.txt', 'w') as f:
for proxy in data['data']:
    print(proxy['ip'], proxy['port'])
        # f.write(f"{proxy['ip']}:{proxy['port']}\n")
print(data)
