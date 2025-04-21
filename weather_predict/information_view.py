import pandas as pd
import matplotlib.pyplot as plt
import warnings
import datetime
warnings.filterwarnings('ignore')


filepath = r'year3_weather_data.csv'
features = pd.read_csv(filepath)
dates = features.iloc[:, 0].tolist()
times = []
for date in dates:
    time = datetime.datetime.strptime(date,'%Y-%m-%d')
    times.append(time)

# 对画布的初始化
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.style.use('fivethirtyeight')
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(4, 2, figsize=(40, 20))

# ==1== 一日的最低温
ax1.plot(times, features['最低温度/℃'])
# 设置x轴y轴标签和title标题
ax1.set_xlabel('')
ax1.set_ylabel('Low_Temperature')
ax1.set_title('最低温度/℃')

# ==2== 一日的最高温
ax2.plot(times, features['最高温度/℃'])
# 设置x轴y轴标签和title标题
ax2.set_xlabel('')
ax2.set_ylabel('High_Temperature')
ax2.set_title('最高温度/℃')

# ==3== 湿度
ax3.plot(times, features['湿度/%'])
# 设置x轴y轴标签和title标题
ax3.set_xlabel('')
ax3.set_ylabel('Humidity')
ax3.set_title('湿度/%')

# # ==4== 风向柱状图实例
# wind_direction_counts = features['风向'].value_counts().sort_index()
# ax4.bar(wind_direction_counts.index, wind_direction_counts.values, color='skyblue')
# for i, v in enumerate(wind_direction_counts.values):
#     ax4.text(i, v + 1, str(v), ha='center', va='bottom')
# # 设置x轴y轴标签和title标题
# ax4.set_xlabel('风向')
# ax4.set_ylabel('天数')
# ax4.set_title('风向分布')

# ==4== 风向
ax4.plot(times, features['风向'])
# 设置x轴y轴标签和title标题
ax5.set_xlabel('')
ax5.set_ylabel('Wind_Direction')
ax5.set_title('风向')

# ==5== 风力
ax5.plot(times, features['风力'])
# 设置x轴y轴标签和title标题
ax5.set_xlabel('')
ax5.set_ylabel('Wind_Level')
ax5.set_title('风力')

# ==6== 紫外线强度
ax6.plot(times, features['紫外线'])
# 设置x轴y轴标签和title标题
ax6.set_xlabel('')
ax6.set_ylabel('Urays')
ax6.set_title('紫外线')

# ==7== 空气质量
ax7.plot(times, features['空气质量'])
# 设置x轴y轴标签和title标题
ax7.set_xlabel('')
ax7.set_ylabel('Air_Quality')
ax7.set_title('空气质量')
# ==8== 置空
ax8.axis('off')
# 轻量化布局调整绘图
plt.tight_layout(pad=2)
plt.show()

