import os

import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model


class WeatherPredictor:
    def __init__(self, model_dir):
        # 设置模型和预处理组件的路径
        model_path = os.path.join(model_dir, 'weather_model.h5')
        scaler_X_path = os.path.join(model_dir, 'scaler_X.save')
        scaler_y_path = os.path.join(model_dir, 'scaler_y.save')
        wind_dir_encoder_path = os.path.join(model_dir, 'wind_dir_encoder.save')
        uv_encoder_path = os.path.join(model_dir, 'uv_encoder.save')

        # 加载预处理组件
        self.model = load_model(model_path)
        self.scaler_X = joblib.load(scaler_X_path)
        self.scaler_y = joblib.load(scaler_y_path)
        self.wind_dir_encoder = joblib.load(wind_dir_encoder_path)
        self.uv_encoder = joblib.load(uv_encoder_path)

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

        wind_dir = self.wind_dir_encoder.inverse_transform([np.argmax(predictions[3][0])])[0]
        wind_force = np.argmax(predictions[4][0])
        uv = self.uv_encoder.inverse_transform([np.argmax(predictions[5][0])])[0]

        return {
            '最低温度': numerical[0][0],
            '最高温度': numerical[0][1],
            '湿度': numerical[0][2],
            '风向': wind_dir,
            '风力': int(wind_force),
            '紫外线': uv,
            '空气质量': numerical[0][3]
        }


def predict_weather(model_dir, date_str):
    predictor = WeatherPredictor(model_dir)
    result = predictor.predict(date_str)
    return result

# # 使用示例
# if __name__ == "__main__":
#     predictor = WeatherPredictor()
#     print("请输入要预测的日期，格式为xxxx-xx-xx样式。")
#     date = input()
#     result = predictor.predict('{date}'.format(date=date))
#     for k, v in result.items():
#         print(f"{k}: {v}")
