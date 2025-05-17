import pandas as pd

position1 = "./day7_weather_data.csv"
# position2 = "./year5_weather_data.csv"
df = pd.read_csv(position1)
data = df.sort_values(by="时间", ascending=True)
data.to_csv("day7_weather_data.csv", mode='w', index=False)


# df = pd.read_csv(position2, encoding='gbk')
# data = df.sort_values(by="时间", ascending=True)
# data.to_csv("year5_weather_data.csv", mode='w', index=False)