import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from datetime import datetime
import matplotlib.pyplot as plt


# 配置参数
class Config:
    seq_len = 5  # 时间窗口长度
    embed_dim = 32  # 嵌入维度
    hidden_dim = 64  # 隐藏层维度
    dropout = 0.4  # Dropout比例
    batch_size = 128  # 批大小
    epochs = 100  # 训练轮次
    lr = 5e-4  # 学习率
    patience = 10  # 早停耐心值


# 新增时间序列窗口函数
def create_sequences(data, targets, window_size=5):
    """将数据转换为LSTM需要的时间序列格式"""
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(targets[i + window_size])
    return np.array(X), np.array(y)


# 数据集类
class WeatherDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.from_numpy(features).float()
        self.targets = {
            'min_temp': torch.from_numpy(targets[:, 0]).float(),
            'max_temp': torch.from_numpy(targets[:, 1]).float(),
            'humidity': torch.from_numpy(targets[:, 2]).float(),
            'wind_dir': torch.from_numpy(targets[:, 3]).long(),
            'wind_force': torch.from_numpy(targets[:, 4]).long(),
            'uv': torch.from_numpy(targets[:, 5]).long(),
            'air_quality': torch.from_numpy(targets[:, 6]).float()
        }

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'targets': {key: value[idx] for key, value in self.targets.items()}
        }


# 模型定义
class WeatherModel(nn.Module):
    def __init__(self, input_dim, cat_sizes):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=Config.hidden_dim,
                            num_layers=3,
                            batch_first=True,
                            dropout=Config.dropout,
                            bidirectional=True)

        # 使用BatchNorm层增强模型稳定性
        self.bn = nn.BatchNorm1d(Config.hidden_dim * 2)

        # 回归任务的全连接层
        self.fc_min_temp = nn.Sequential(
            nn.Linear(Config.hidden_dim * 2, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 1)
        )

        self.fc_max_temp = nn.Sequential(
            nn.Linear(Config.hidden_dim * 2, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 1)
        )

        self.fc_humidity = nn.Sequential(
            nn.Linear(Config.hidden_dim * 2, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 1)
        )

        self.fc_air_quality = nn.Sequential(
            nn.Linear(Config.hidden_dim * 2, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 1)
        )

        # 分类任务的全连接层
        self.fc_wind_dir = nn.Sequential(
            nn.Linear(Config.hidden_dim * 2, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, cat_sizes['wind_dir'])
        )

        self.fc_wind_force = nn.Sequential(
            nn.Linear(Config.hidden_dim * 2, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, cat_sizes['wind_force'])
        )

        self.fc_uv = nn.Sequential(
            nn.Linear(Config.hidden_dim * 2, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, cat_sizes['uv'])
        )

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # 取序列最后一个时间步的输出

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


def train():
    city = 'beijing'

    # 1. 数据加载与特征工程
    df = pd.read_csv(r'../combined_file.csv', encoding='utf-8')

    # 时间特征处理（保持不变）
    df['时间'] = pd.to_datetime(df['时间'])
    df['年'] = df['时间'].dt.year
    df['月'] = df['时间'].dt.month
    df['日'] = df['时间'].dt.day
    seasons = {1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2,
               7: 2, 8: 2, 9: 3, 10: 3, 11: 3, 12: 0}
    df['季节'] = df['月'].map(seasons)

    # 2. 特征标准化
    X = df[['年', '月', '日', '季节']]
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    # 3. 目标变量处理（保持不变）
    wind_dir_encoder = LabelEncoder()
    uv_encoder = LabelEncoder()
    df['风向'] = wind_dir_encoder.fit_transform(df['风向'])
    uv_labels = ['很强', '强', '中等', '弱', '最弱']
    uv_encoder.fit(uv_labels)
    df['紫外线'] = uv_encoder.transform(df['紫外线'])
    y = df[['最低温度', '最高温度', '湿度', '风向', '风力', '紫外线', '空气质量']]
    numerical_cols = ['最低温度', '最高温度', '湿度', '空气质量']
    scaler_y = StandardScaler()
    y[numerical_cols] = scaler_y.fit_transform(y[numerical_cols])

    # 4. 创建时间序列数据集
    X_sequences, y_sequences = create_sequences(X_scaled, y.values, Config.seq_len)

    # 5. 数据分割（注意保持时间顺序）
    split = int(0.8 * len(X_sequences))
    X_train, X_test = X_sequences[:split], X_sequences[split:]
    y_train, y_test = y_sequences[:split], y_sequences[split:]

    # 6. 数据加载器
    train_dataset = WeatherDataset(X_train, y_train)
    test_dataset = WeatherDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size)

    # 7. 模型初始化
    model = WeatherModel(input_dim=X.shape[1], cat_sizes={
        'wind_dir': len(wind_dir_encoder.classes_),
        'wind_force': 7,
        'uv': len(uv_encoder.classes_)
    })

    # 8. 定义损失函数和优化器
    criterion = {
        'min_temp': nn.MSELoss(),
        'max_temp': nn.MSELoss(),
        'humidity': nn.MSELoss(),
        'wind_dir': nn.CrossEntropyLoss(),
        'wind_force': nn.CrossEntropyLoss(),
        'uv': nn.CrossEntropyLoss(),
        'air_quality': nn.MSELoss()
    }
    optimizer = optim.AdamW(model.parameters(), lr=Config.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # 9. 训练模型
    best_loss = float('inf')
    patience_counter = 0
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    for epoch in range(Config.epochs):
        model.train()
        train_loss = 0
        correct_wind_dir = 0
        correct_wind_force = 0
        correct_uv = 0
        total = 0

        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(batch['features'])
            loss = sum([criterion[key](outputs[key], batch['targets'][key]) for key in criterion])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # 计算分类任务准确率
            _, predicted_wind_dir = outputs['wind_dir'].max(1)
            _, predicted_wind_force = outputs['wind_force'].max(1)
            _, predicted_uv = outputs['uv'].max(1)

            total += batch['targets']['wind_dir'].size(0)
            correct_wind_dir += predicted_wind_dir.eq(batch['targets']['wind_dir']).sum().item()
            correct_wind_force += predicted_wind_force.eq(batch['targets']['wind_force']).sum().item()
            correct_uv += predicted_uv.eq(batch['targets']['uv']).sum().item()

        # 计算训练准确率
        train_acc = {
            'wind_dir': 100. * correct_wind_dir / total,
            'wind_force': 100. * correct_wind_force / total,
            'uv': 100. * correct_uv / total
        }
        train_accs.append(train_acc)

        model.eval()
        test_loss = 0
        correct_wind_dir = 0
        correct_wind_force = 0
        correct_uv = 0
        total = 0

        with torch.no_grad():
            for batch in test_loader:
                outputs = model(batch['features'])
                loss = sum([criterion[key](outputs[key], batch['targets'][key]) for key in criterion])
                test_loss += loss.item()

                # 计算分类任务准确率
                _, predicted_wind_dir = outputs['wind_dir'].max(1)
                _, predicted_wind_force = outputs['wind_force'].max(1)
                _, predicted_uv = outputs['uv'].max(1)

                total += batch['targets']['wind_dir'].size(0)
                correct_wind_dir += predicted_wind_dir.eq(batch['targets']['wind_dir']).sum().item()
                correct_wind_force += predicted_wind_force.eq(batch['targets']['wind_force']).sum().item()
                correct_uv += predicted_uv.eq(batch['targets']['uv']).sum().item()

        # 计算测试准确率
        test_acc = {
            'wind_dir': 100. * correct_wind_dir / total,
            'wind_force': 100. * correct_wind_force / total,
            'uv': 100. * correct_uv / total
        }
        test_accs.append(test_acc)

        avg_train_loss = train_loss / len(train_loader)
        avg_test_loss = test_loss / len(test_loader)
        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)

        scheduler.step(avg_test_loss)

        # 早停
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= Config.patience:
                print(f"Early stopping at epoch {epoch}")
                break

        print(f'Epoch {epoch + 1}: Train Loss={avg_train_loss:.4f}, Test Loss={avg_test_loss:.4f}')
        print(f'Train Acc - Wind Dir: {train_acc["wind_dir"]:.2f}% | Wind Force: {train_acc["wind_force"]:.2f}% | UV: {train_acc["uv"]:.2f}%')
        print(f'Test Acc - Wind Dir: {test_acc["wind_dir"]:.2f}% | Wind Force: {test_acc["wind_force"]:.2f}% | UV: {test_acc["uv"]:.2f}%')

    # 加载最佳模型
    model.load_state_dict(torch.load('best_model.pth', weights_only=True))

    # 10. 可视化训练过程
    plt.figure(figsize=(12, 8))

    # 绘制损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制分类准确率曲线
    plt.subplot(2, 2, 2)
    plt.plot([acc['wind_dir'] for acc in train_accs], label='Wind Dir Train Acc')
    plt.plot([acc['wind_dir'] for acc in test_accs], label='Wind Dir Test Acc')
    plt.plot([acc['wind_force'] for acc in train_accs], label='Wind Force Train Acc')
    plt.plot([acc['wind_force'] for acc in test_accs], label='Wind Force Test Acc')
    plt.plot([acc['uv'] for acc in train_accs], label='UV Train Acc')
    plt.plot([acc['uv'] for acc in test_accs], label='UV Test Acc')
    plt.title('Classification Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # 11. 预测函数改造
    def predict_weather(date_str):
        """需要提供连续seq_len天的数据"""
        # 获取目标日期前seq_len天的数据
        target_date = pd.to_datetime(date_str)
        start_date = target_date - pd.Timedelta(days=Config.seq_len)

        # 构造输入序列
        mask = (df['时间'] >= start_date) & (df['时间'] < target_date)
        last_n_days = df.loc[mask].tail(Config.seq_len)

        if len(last_n_days) < Config.seq_len:
            raise ValueError(f"需要至少{Config.seq_len}天的历史数据")

        # 预处理
        X_new = last_n_days[['年', '月', '日', '季节']].values
        X_new_scaled = scaler_X.transform(X_new)

        # 预测
        model.eval()
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

        wind_dir_prob_dict = {wind_dir_labels[i]: f"{wind_dir_probs[0, i].item() * 100:.2f}%" for i in
                              range(len(wind_dir_labels))}
        wind_force_prob_dict = {f"Force {wind_force_labels[i]}": f"{wind_force_probs[0, i].item() * 100:.2f}%" for i in
                                range(len(wind_force_labels))}
        uv_prob_dict = {uv_labels[i]: f"{uv_probs[0, i].item() * 100:.2f}%" for i in range(len(uv_labels))}

        return {
            '最低温度': numerical[0],
            '最高温度': numerical[1],
            '湿度': numerical[2],
            # '风向': wind_dir_encoder.inverse_transform([wind_dir_pred])[0],
            # '风向概率': f"{wind_dir_prob:.2f}%",
            '风向概率分布': wind_dir_prob_dict,
            # '风力': wind_force_pred,
            # '风力概率': f"{wind_force_prob:.2f}%",
            '风力概率分布': wind_force_prob_dict,
            # '紫外线': uv_encoder.inverse_transform([uv_pred])[0],
            # '紫外线概率': f"{uv_prob:.2f}%",
            '紫外线概率分布': uv_prob_dict,
            '空气质量': numerical[3]
        }

    # try:
    #     prediction = predict_weather('2024-6-29')
    #     for k, v in prediction.items():
    #         print(f'{k}: {v}')
    # except ValueError as e:
    #     print(f"预测失败: {e}")

    # 保存模型和预处理组件
    torch.save(model.state_dict(), f'../pytorch_model/{city}/weather_model_lstm.pth')
    joblib.dump(scaler_X, f'../pytorch_model/{city}/scaler_X.save')
    joblib.dump(scaler_y, f'../pytorch_model/{city}/scaler_y.save')
    joblib.dump(wind_dir_encoder, f'../pytorch_model/{city}/wind_dir_encoder.save')
    joblib.dump(uv_encoder, f'../pytorch_model/{city}/uv_encoder.save')

    print("LSTM模型和预处理组件已保存")


if __name__ == '__main__':
    train()