import joblib
import torch
import numpy as np
from datetime import datetime
import pandas as pd

# 配置参数
class Config:
    seq_len = 5  # 时间窗口长度
    embed_dim = 32  # 嵌入维度
    hidden_dim = 64  # 隐藏层维度
    dropout = 0.4  # Dropout比例

# 模型定义
class WeatherModel(torch.nn.Module):
    def __init__(self, input_dim, cat_sizes):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=input_dim, hidden_size=Config.hidden_dim,
                                  num_layers=3,
                                  batch_first=True,
                                  dropout=Config.dropout,
                                  bidirectional=True)

        self.bn = torch.nn.BatchNorm1d(Config.hidden_dim * 2)

        self.fc_min_temp = torch.nn.Sequential(
            torch.nn.Linear(Config.hidden_dim * 2, 32),
            torch.nn.GELU(),
            torch.nn.Linear(32, 16),
            torch.nn.GELU(),
            torch.nn.Linear(16, 1)
        )

        self.fc_max_temp = torch.nn.Sequential(
            torch.nn.Linear(Config.hidden_dim * 2, 32),
            torch.nn.GELU(),
            torch.nn.Linear(32, 16),
            torch.nn.GELU(),
            torch.nn.Linear(16, 1)
        )

        self.fc_humidity = torch.nn.Sequential(
            torch.nn.Linear(Config.hidden_dim * 2, 32),
            torch.nn.GELU(),
            torch.nn.Linear(32, 16),
            torch.nn.GELU(),
            torch.nn.Linear(16, 1)
        )

        self.fc_air_quality = torch.nn.Sequential(
            torch.nn.Linear(Config.hidden_dim * 2, 32),
            torch.nn.GELU(),
            torch.nn.Linear(32, 16),
            torch.nn.GELU(),
            torch.nn.Linear(16, 1)
        )

        self.fc_wind_dir = torch.nn.Sequential(
            torch.nn.Linear(Config.hidden_dim * 2, 32),
            torch.nn.GELU(),
            torch.nn.Linear(32, 16),
            torch.nn.GELU(),
            torch.nn.Linear(16, cat_sizes['wind_dir'])
        )

        self.fc_wind_force = torch.nn.Sequential(
            torch.nn.Linear(Config.hidden_dim * 2, 32),
            torch.nn.GELU(),
            torch.nn.Linear(32, 16),
            torch.nn.GELU(),
            torch.nn.Linear(16, cat_sizes['wind_force'])
        )

        self.fc_uv = torch.nn.Sequential(
            torch.nn.Linear(Config.hidden_dim * 2, 32),
            torch.nn.GELU(),
            torch.nn.Linear(32, 16),
            torch.nn.GELU(),
            torch.nn.Linear(16, cat_sizes['uv'])
        )

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.bn(x)

        min_temp = self.fc_min_temp(x)
        max_temp = self.fc_max_temp(x)
        humidity = self.fc_humidity(x)
        wind_dir = self.fc_wind_dir(x)
        wind_force = self.fc_wind_force(x)
        uv = self.fc_uv(x)
        air_quality = self.fc_air_quality(x)

        return {
            'min_temp': min_temp.squeeze(),
            'max_temp': max_temp.squeeze(),
            'humidity': humidity.squeeze(),
            'wind_dir': wind_dir,
            'wind_force': wind_force,
            'uv': uv,
            'air_quality': air_quality.squeeze()
        }

def predict_weather(date_str, model_path, scaler_X_path, scaler_y_path, wind_dir_encoder_path, uv_encoder_path):
    # 加载模型和预处理组件
    scaler_X = joblib.load(scaler_X_path)
    scaler_y = joblib.load(scaler_y_path)
    wind_dir_encoder = joblib.load(wind_dir_encoder_path)
    uv_encoder = joblib.load(uv_encoder_path)

    # 读取数据
    df = pd.read_csv(r'F:\Rayedata2\weather_crawl\beijing_year5_weather_data.csv', encoding='utf-8')
    df['时间'] = pd.to_datetime(df['时间'])
    df['年'] = df['时间'].dt.year
    df['月'] = df['时间'].dt.month
    df['日'] = df['时间'].dt.day
    seasons = {1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3, 12: 0}
    df['季节'] = df['月'].map(seasons)

    # 获取目标日期前seq_len天的数据
    target_date = pd.to_datetime(date_str)
    start_date = target_date - pd.Timedelta(days=Config.seq_len)
    mask = (df['时间'] >= start_date) & (df['时间'] < target_date)
    last_n_days = df.loc[mask].tail(Config.seq_len)

    if len(last_n_days) < Config.seq_len:
        raise ValueError(f"需要至少{Config.seq_len}天的历史数据")

    # 预处理
    X_new = last_n_days[['年', '月', '日', '季节']].values
    X_new_scaled = scaler_X.transform(X_new)

    # 初始化并加载模型
    model = WeatherModel(input_dim=X_new.shape[1], cat_sizes={
        'wind_dir': len(wind_dir_encoder.classes_),
        'wind_force': 7,
        'uv': len(uv_encoder.classes_)
    })
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    # 预测
    with torch.no_grad():
        predictions = model(torch.tensor(X_new_scaled).unsqueeze(0).float())

    # 后处理
    min_temp = predictions['min_temp'].item()
    max_temp = predictions['max_temp'].item()
    humidity = predictions['humidity'].item()
    air_quality = predictions['air_quality'].item()

    # 对数值特征进行逆归一化
    numerical = np.array([[min_temp, max_temp, humidity, air_quality]]).reshape(1, -1)
    numerical = scaler_y.inverse_transform(numerical).flatten()

    # 获取分类预测的概率分布
    wind_dir_probs = torch.nn.functional.softmax(predictions['wind_dir'], dim=1)
    wind_force_probs = torch.nn.functional.softmax(predictions['wind_force'], dim=1)
    uv_probs = torch.nn.functional.softmax(predictions['uv'], dim=1)

    # 获取分类预测的类别
    wind_dir_pred = torch.argmax(wind_dir_probs).item()
    wind_force_pred = torch.argmax(wind_force_probs).item()
    uv_pred = torch.argmax(uv_probs).item()

    # 获取分类预测的概率（百分比格式）
    wind_dir_prob = wind_dir_probs[0, wind_dir_pred].item() * 100
    wind_force_prob = wind_force_probs[0, wind_force_pred].item() * 100
    uv_prob = uv_probs[0, uv_pred].item() * 100

    # 将概率分布转换为字典形式（百分比格式）
    wind_dir_labels = wind_dir_encoder.classes_
    wind_force_labels = [0, 1, 2, 3, 4, 5, 6]  # 根据实际类别调整
    uv_labels = uv_encoder.classes_

    wind_dir_prob_dict = {wind_dir_labels[i]: f"{wind_dir_probs[0, i].item() * 100:.2f}%" for i in range(len(wind_dir_labels))}
    wind_force_prob_dict = {f"Force {wind_force_labels[i]}": f"{wind_force_probs[0, i].item() * 100:.2f}%" for i in range(len(wind_force_labels))}
    uv_prob_dict = {uv_labels[i]: f"{uv_probs[0, i].item() * 100:.2f}%" for i in range(len(uv_labels))}

    return {
        '最低温度': numerical[0],
        '最高温度': numerical[1],
        '湿度': numerical[2],
        '风向': wind_dir_encoder.inverse_transform([wind_dir_pred])[0],
        # '风向概率': f"{wind_dir_prob:.2f}%",
        '风向概率分布': wind_dir_prob_dict,
        '风力': wind_force_pred,
        # '风力概率': f"{wind_force_prob:.2f}%",
        '风力概率分布': wind_force_prob_dict,
        '紫外线': uv_encoder.inverse_transform([uv_pred])[0],
        # '紫外线概率': f"{uv_prob:.2f}%",
        '紫外线概率分布': uv_prob_dict,
        '空气质量': numerical[3]
    }


if __name__ == '__main__':
    model_path = '../pytorch_model/beijing/weather_model_lstm.pth'
    scaler_X_path = '../pytorch_model/beijing/scaler_X.save'
    scaler_y_path = '../pytorch_model/beijing/scaler_y.save'
    wind_dir_encoder_path = '../pytorch_model/beijing/wind_dir_encoder.save'
    uv_encoder_path = '../pytorch_model/beijing/uv_encoder.save'

    try:
        prediction = predict_weather('2024-06-29', model_path, scaler_X_path, scaler_y_path, wind_dir_encoder_path, uv_encoder_path)
        for k, v in prediction.items():
            if isinstance(v, dict):
                print(f"{k}:")
                for sub_k, sub_v in v.items():
                    print(f"  {sub_k}: {sub_v}")
            else:
                print(f"{k}: {v}")
    except ValueError as e:
        print(f"预测失败: {e}")