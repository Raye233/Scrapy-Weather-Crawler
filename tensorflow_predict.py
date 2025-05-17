import os

import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

city = 'wuhan'
path = os.path.join(r'F:\Rayedata2\weather_crawl\tensorflow_model', city)

class WeatherPredictor:
    def __init__(self):
        # 加载所有组件
        self.model = load_model(f'{path}/weather_model.h5')
        self.scaler_X = joblib.load(f'{path}/scaler_X.save')
        self.scaler_y = joblib.load(f'{path}/scaler_y.save')
        self.wind_dir_encoder = joblib.load(f'{path}/wind_dir_encoder.save')
        self.uv_encoder = joblib.load(f'{path}/uv_encoder.save')

        # 季节映射（需要与训练时一致）
        self.seasons = {1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2,
                        7: 2, 8: 2, 9: 3, 10: 3, 11: 3, 12: 0}

    def predict(self, date_str):
        # 创建测试数据
        date_df = pd.DataFrame({'时间': [date_str]})
        date_df['时间'] = pd.to_datetime(date_df['时间'])
        date_df['年'] = date_df['时间'].dt.year
        date_df['月'] = date_df['时间'].dt.month
        date_df['日'] = date_df['时间'].dt.day
        date_df['季节'] = date_df['月'].map(self.seasons)

        X_new = date_df[['年', '月', '日', '季节']]
        X_new_scaled = self.scaler_X.transform(X_new)

        # 进行预测
        predictions = self.model.predict(X_new_scaled)

        # 处理预测结果
        numerical = np.array([[predictions[0][0][0], predictions[1][0][0],
                               predictions[2][0][0], predictions[6][0][0]]])
        numerical = self.scaler_y.inverse_transform(numerical)

        # 计算分类任务的概率分布
        wind_dir_probs = predictions[3][0]
        wind_force_probs = predictions[4][0]
        uv_probs = predictions[5][0]

        # 获取分类预测的类别
        wind_dir_pred = np.argmax(wind_dir_probs)
        wind_force_pred = np.argmax(wind_force_probs)
        uv_pred = np.argmax(uv_probs)

        # 将概率分布转换为字典形式（百分比格式）
        wind_dir_labels = self.wind_dir_encoder.classes_
        wind_force_labels = [0, 1, 2, 3, 4, 5, 6]  # 根据实际类别调整
        uv_labels = self.uv_encoder.classes_

        wind_dir_prob_dict = {wind_dir_labels[i]: f"{wind_dir_probs[i] * 100:.2f}%" for i in
                              range(len(wind_dir_labels))}
        wind_force_prob_dict = {f"Force {wind_force_labels[i]}": f"{wind_force_probs[i] * 100:.2f}%" for i in
                                range(len(wind_force_labels))}
        uv_prob_dict = {uv_labels[i]: f"{uv_probs[i] * 100:.2f}%" for i in range(len(uv_labels))}

        # wind_dir = self.wind_dir_encoder.inverse_transform([np.argmax(predictions[3][0])])[0]
        # wind_force = np.argmax(predictions[4][0])
        # uv = self.uv_encoder.inverse_transform([np.argmax(predictions[5][0])])[0]

        return {
            '最低温度': numerical[0][0],
            '最高温度': numerical[0][1],
            '湿度': numerical[0][2],
            '风向': self.wind_dir_encoder.inverse_transform([wind_dir_pred])[0],
            '风向概率分布': wind_dir_prob_dict,
            '风力': int(wind_force_pred),
            '风力概率分布': wind_force_prob_dict,
            '紫外线': self.uv_encoder.inverse_transform([uv_pred])[0],
            '紫外线概率分布': uv_prob_dict,
            '空气质量': numerical[0][3]
        }


# 使用示例
if __name__ == "__main__":
    predictor = WeatherPredictor()
    print("请输入要预测的日期，格式为xxxx-xx-xx样式。")
    date = input()
    result = predictor.predict('{date}'.format(date=date))
    for k, v in result.items():
        if isinstance(v, dict):
            print(f"{k}:")
            for sub_k, sub_v in v.items():
                print(f"  {sub_k}: {sub_v}")
        else:
            print(f"{k}: {v}")
