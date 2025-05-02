import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from datetime import datetime


# 配置参数
class Config:
    seq_len = 7  # 时间窗口长度
    embed_dim = 8  # 嵌入维度
    hidden_dim = 512  # 隐藏层维度
    dropout = 0.3  # Dropout比例
    batch_size = 64  # 批大小
    epochs = 200  # 训练轮次
    lr = 2e-4  # 学习率
    patience = 15  # 早停耐心值


VALID_WIND_DIRECTIONS = ['东风', '南风', '西风', '北风', '东南风', '东北风', '西南风', '西北风']
VALID_UV_LEVELS = ['最弱', '弱', '中等', '强', '很强']
VALID_WIND_POWER = list(range(7))  # 0-6级


class WeatherDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            'features': torch.tensor(self.features[idx], dtype=torch.float32),
            'targets': {
                'min_temp': torch.tensor(float(self.targets['min_temp'][idx])),
                'max_temp': torch.tensor(float(self.targets['max_temp'][idx])),
                'humidity': torch.tensor(float(self.targets['humidity'][idx])),
                'wind_dir': torch.tensor(int(self.targets['wind_dir'][idx]), dtype=torch.long),
                'wind_power': torch.tensor(int(self.targets['wind_power'][idx]), dtype=torch.long),
                'uv': torch.tensor(int(self.targets['uv'][idx]), dtype=torch.long),
                'air_quality': torch.tensor(float(self.targets['air_quality'][idx]))
            }
        }


class WeatherModel(nn.Module):
    def __init__(self, num_numerical_features, cat_sizes):
        super().__init__()
        self.num_numerical = num_numerical_features
        self.num_categorical = len(cat_sizes)

        # 嵌入层
        self.embeddings = nn.ModuleDict({
            'wind_dir': nn.Embedding(cat_sizes['wind_dir'], Config.embed_dim),
            'uv': nn.Embedding(cat_sizes['uv'], Config.embed_dim // 2),
            'wind_power': nn.Embedding(cat_sizes['wind_power'], Config.embed_dim // 2)
        })

        total_input_dim = num_numerical_features + (Config.embed_dim + (Config.embed_dim // 2) * 2)

        # 特征融合
        self.feature_fusion = nn.Sequential(
            nn.Linear(total_input_dim, Config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(Config.dropout),
            nn.Linear(Config.hidden_dim, Config.hidden_dim),
            nn.ReLU()
        )

        # 时序处理
        self.temporal_layer = nn.LSTM(
            input_size=Config.hidden_dim,
            hidden_size=Config.hidden_dim,
            batch_first=True
        )

        # 多任务输出
        self.output_heads = nn.ModuleDict({
            'min_temp': nn.Linear(Config.hidden_dim, 1),
            'max_temp': nn.Linear(Config.hidden_dim, 1),
            'humidity': nn.Linear(Config.hidden_dim, 1),
            'wind_dir': nn.Linear(Config.hidden_dim, cat_sizes['wind_dir']),
            'wind_power': nn.Linear(Config.hidden_dim, cat_sizes['wind_power']),
            'uv': nn.Linear(Config.hidden_dim, cat_sizes['uv']),
            'air_quality': nn.Linear(Config.hidden_dim, 1)
        })

    def forward(self, x):
        numerical = x[:, :, :self.num_numerical]
        categorical = x[:, :, self.num_numerical:].long()

        # 安全处理分类特征
        categorical[:, :, 0] = torch.clamp(categorical[:, :, 0], 0, len(VALID_WIND_DIRECTIONS) - 1)
        categorical[:, :, 1] = torch.clamp(categorical[:, :, 1], 0, len(VALID_UV_LEVELS) - 1)
        categorical[:, :, 2] = torch.clamp(categorical[:, :, 2], 0, len(VALID_WIND_POWER) - 1)

        # 嵌入处理
        wind_emb = self.embeddings['wind_dir'](categorical[..., 0])
        uv_emb = self.embeddings['uv'](categorical[..., 1])
        wp_emb = self.embeddings['wind_power'](categorical[..., 2])

        combined = torch.cat([numerical, wind_emb, uv_emb, wp_emb], dim=-1)
        fused = self.feature_fusion(combined)
        temporal, _ = self.temporal_layer(fused)
        temporal = temporal[:, -1, :]

        outputs = {name: head(temporal) for name, head in self.output_heads.items()}
        return outputs


def preprocess_data(raw_df):
    df = raw_df.copy()

    # 数据清洗
    df['紫外线'] = df['紫外线'].str.replace(r'[^a-zA-Z\u4e00-\u9fa5]', '强', regex=True)
    df['风力'] = pd.to_numeric(df['风力'], errors='coerce')

    # 过滤非法数据
    df = df[
        df['风向'].isin(VALID_WIND_DIRECTIONS) &
        df['紫外线'].isin(VALID_UV_LEVELS) &
        df['风力'].between(0, 6)
        ].dropna()

    # 分类编码
    encoders = {
        'wind_dir': LabelEncoder().fit(VALID_WIND_DIRECTIONS),
        'uv': LabelEncoder().fit(VALID_UV_LEVELS),
        'wind_power': LabelEncoder().fit(VALID_WIND_POWER)
    }
    df['风向编码'] = encoders['wind_dir'].transform(df['风向'])
    df['紫外线编码'] = encoders['uv'].transform(df['紫外线'])
    df['风力编码'] = df['风力'].astype(int)

    # 验证编码范围
    assert df['风向编码'].between(0, 7).all(), "风向编码值越界 0-7"
    assert df['紫外线编码'].between(0, 4).all(), "紫外线编码值越界 0-4"
    assert df['风力编码'].between(0, 6).all(), "风力编码值越界 0-6"

    # 时间特征
    df['时间'] = pd.to_datetime(df['时间'])
    df['年'] = df['时间'].dt.year
    df['月'] = df['时间'].dt.month
    df['日'] = df['时间'].dt.day
    df['季节'] = (df['月'] % 12 + 3) // 3

    # 滞后特征
    for i in range(1, 4):
        df[f'滞后{i}_最低温度'] = df['最低温度'].shift(i)
        df[f'滞后{i}_最高温度'] = df['最高温度'].shift(i)
    df = df.dropna()

    # 特征标准化
    numerical_features = ['年', '月', '日', '季节', '滞后1_最低温度', '滞后1_最高温度']
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    # 构造序列数据
    sequences = []
    targets = []
    feature_columns = numerical_features + ['风向编码', '紫外线编码', '风力编码']

    for i in range(len(df) - Config.seq_len):
        seq = df.iloc[i:i + Config.seq_len][feature_columns].values

        # 目标特征顺序验证
        tgt_row = df.iloc[i + Config.seq_len][
            ['最低温度', '最高温度', '湿度', '风向编码', '风力编码', '紫外线编码', '空气质量']
        ]
        targets.append([
            float(tgt_row['最低温度']),
            float(tgt_row['最高温度']),
            float(tgt_row['湿度']),
            int(tgt_row['风向编码']),
            int(tgt_row['风力编码']),
            int(tgt_row['紫外线编码']),
            float(tgt_row['空气质量'])
        ])
        sequences.append(seq)

    # 转换为结构化数组
    targets = np.core.records.fromarrays(
        list(zip(*targets)),
        dtype=[
            ('min_temp', 'f4'),
            ('max_temp', 'f4'),
            ('humidity', 'f4'),
            ('wind_dir', 'i8'),
            ('wind_power', 'i8'),
            ('uv', 'i8'),
            ('air_quality', 'f4')
        ]
    )
    sequences = np.array(sequences, dtype=np.float32)

    print(f"数据验证 - 序列形状: {sequences.shape}, 目标类型: {type(targets)}")
    return df, sequences, targets, scaler, encoders


def train_model(model, train_loader, val_loader):
    optimizer = optim.AdamW(model.parameters(), lr=Config.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    criterion = {
        'min_temp': nn.MSELoss(),
        'max_temp': nn.MSELoss(),
        'humidity': nn.MSELoss(),
        'wind_dir': nn.CrossEntropyLoss(),
        'wind_power': nn.CrossEntropyLoss(),
        'uv': nn.CrossEntropyLoss(),
        'air_quality': nn.MSELoss()
    }

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(Config.epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(batch['features'])

            loss = 0
            for name in criterion:
                pred = outputs[name].squeeze() if name in ['min_temp', 'max_temp', 'humidity', 'air_quality'] else \
                outputs[name]
                tgt = batch['targets'][name]
                loss += criterion[name](pred, tgt) * (1.0 if 'temp' in name else 0.8)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(batch['features'])
                batch_loss = 0
                for name in criterion:
                    pred = outputs[name].squeeze() if name in ['min_temp', 'max_temp', 'humidity', 'air_quality'] else \
                    outputs[name]
                    tgt = batch['targets'][name]
                    batch_loss += criterion[name](pred, tgt)
                val_loss += batch_loss.item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        # 早停机制
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= Config.patience:
                print(f"Early stopping at epoch {epoch}")
                break

        print(
            f'Epoch {epoch + 1:03d} | Train Loss: {train_loss / len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f}')

    model.load_state_dict(torch.load('best_model.pth'))
    return model


def predict(model, date_str, processed_df, scaler, encoders):
    date = datetime.strptime(date_str, '%Y/%m/%d')
    last_data = processed_df.iloc[-3]  # 使用倒数第三天的数据

    features = {
        '年': date.year,
        '月': date.month,
        '日': date.day,
        '季节': (date.month % 12 + 3) // 3,
        '滞后1_最低温度': last_data['最低温度'],
        '滞后1_最高温度': last_data['最高温度'],
        '风向编码': last_data['风向编码'],
        '紫外线编码': last_data['紫外线编码'],
        '风力编码': last_data['风力编码']
    }

    # 构建序列
    seq_df = pd.DataFrame([features] * Config.seq_len)
    numerical_features = ['年', '月', '日', '季节', '滞后1_最低温度', '滞后1_最高温度']
    seq_df[numerical_features] = scaler.transform(seq_df[numerical_features])

    feature_columns = numerical_features + ['风向编码', '紫外线编码', '风力编码']
    seq_tensor = torch.tensor(seq_df[feature_columns].values, dtype=torch.float32).unsqueeze(0)

    # 预测
    model.eval()
    with torch.no_grad():
        outputs = model(seq_tensor)

    # 解码结果
    prediction = {
        '最低温度': outputs['min_temp'].item(),
        '最高温度': outputs['max_temp'].item(),
        '湿度': outputs['humidity'].item(),
        '风向': encoders['wind_dir'].inverse_transform([torch.argmax(outputs['wind_dir']).item()])[0],
        '风力': encoders['wind_power'].inverse_transform([torch.argmax(outputs['wind_power']).item()])[0],
        '紫外线': encoders['uv'].inverse_transform([torch.argmax(outputs['uv']).item()])[0],
        '空气质量': outputs['air_quality'].item()
    }
    return prediction


if __name__ == "__main__":
    # 加载数据
    raw_df = pd.read_csv("../combined_file.csv",
                         encoding='utf-8',
                         names=["时间", "最低温度", "最高温度", "湿度", "风向", "风力", "紫外线", "空气质量"],
                         skiprows=1)

    # 数据预处理
    processed_df, sequences, targets, scaler, encoders = preprocess_data(raw_df)

    # 数据验证
    print("\n预处理后数据验证:")
    print(f"- 总样本数: {len(processed_df)}")
    print(f"- 序列形状: {sequences.shape} (应= (样本数, {Config.seq_len}, 9))")
    print(f"- 目标类型: {type(targets)}")
    print(f"- 风向编码范围: {np.min(targets['wind_dir'])}-{np.max(targets['wind_dir'])} (应0-7)")
    print(f"- 空气质量范围: {np.min(targets['air_quality'])}-{np.max(targets['air_quality'])}")

    # 划分数据集
    X_train, X_val, y_train, y_val = train_test_split(
        sequences, targets, test_size=0.2, shuffle=False
    )

    # 创建数据加载器
    train_dataset = WeatherDataset(X_train, y_train)
    val_dataset = WeatherDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size)

    # 初始化模型
    model = WeatherModel(
        num_numerical_features=6,
        cat_sizes={
            'wind_dir': len(encoders['wind_dir'].classes_),
            'uv': len(encoders['uv'].classes_),
            'wind_power': len(encoders['wind_power'].classes_)
        }
    )

    # 训练模型
    trained_model = train_model(model, train_loader, val_loader)

    # # 示例预测
    # try:
    #     sample_pred = predict(trained_model, '2024/12/31', processed_df, scaler, encoders)
    #     print("\n预测结果示例：")
    #     for k, v in sample_pred.items():
    #         print(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}")
    # except Exception as e:
    #     print(f"预测时发生错误：{str(e)}")
