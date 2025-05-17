import pandas as pd

# 读取第一个CSV文件
df1 = pd.read_csv('/weather_predict/weather_project/year5_weather_data.csv', encoding='gbk')

# 读取第二个CSV文件
df2 = pd.read_csv('/weather_predict/weather_project/year3_weather_data.csv', encoding='gbk')

# 合并两个CSV文件
combined_df = pd.concat([df1, df2], ignore_index=True)

# 保存合并后的CSV文件
combined_df.to_csv('combined_file.csv', index=False, encoding='gbk')