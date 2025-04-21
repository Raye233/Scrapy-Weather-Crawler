import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 读取CSV文件
df = pd.read_csv('year3_weather_data.csv')

# 数据预处理
df['时间'] = pd.to_datetime(df['时间'])
df['年'] = df['时间'].dt.year
df['月'] = df['时间'].dt.month
df['日'] = df['时间'].dt.day

label_encoder_wind = LabelEncoder()
df['风向'] = label_encoder_wind.fit_transform(df['风向'])

label_encoder_uv = LabelEncoder()
df['紫外线'] = label_encoder_uv.fit_transform(df['紫外线'])

features = df[['年', '月', '日']].values
labels = df[['最低温度/℃', '最高温度/℃', '湿度/%', '风向', '风力', '紫外线', '空气质量']].values

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# 数据标准化
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

# 转换为PyTorch张量
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # 添加序列维度
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)  # 添加序列维度
y_test = torch.tensor(y_test, dtype=torch.float32)


class LSTMWeatherPredictionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMWeatherPredictionModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# 模型参数
input_size = X_train.shape[2]  # 输入特征维度
hidden_size = 32  # 隐含层神经元个数
num_layers = 3  # LSTM层数
output_size = 7  # 输出维度为7（7个目标变量）

# 实例化模型
model = LSTMWeatherPredictionModel(input_size, hidden_size, num_layers, output_size)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 1000
best_test_loss = float('inf')
patience = 50
patience_counter = 0

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        test_loss = criterion(predictions, y_test)
        print(f'Test Loss: {test_loss.item():.4f}')

    if test_loss < best_test_loss:
        best_test_loss = test_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered")
            break

# 评估模型
model.eval()
with torch.no_grad():
    predictions = model(X_test)
    predictions = scaler_y.inverse_transform(predictions.numpy())
    y_test_original = scaler_y.inverse_transform(y_test.numpy())

    mse = mean_squared_error(y_test_original, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_original, predictions)
    r2 = r2_score(y_test_original, predictions)

    print(f'Mean Squared Error: {mse:.4f}')
    print(f'Root Mean Squared Error: {rmse:.4f}')
    print(f'Mean Absolute Error: {mae:.4f}')
    print(f'R² Score: {r2:.4f}')

# 确保dates只包含测试集对应的日期
test_dates = df['时间'].iloc[-X_test.shape[0]:].dt.to_pydatetime().tolist()

true_data = pd.DataFrame(data={'date': test_dates, 'actual': y_test_original[:, 1]})  # 使用最高温度
predictions_data = pd.DataFrame(data={'date': test_dates, 'prediction': predictions[:, 1]})  # 使用最高温度

plt.figure(figsize=(20, 5))
plt.plot(true_data['date'], true_data['actual'], 'b-', label='actual')
plt.plot(predictions_data['date'], predictions_data['prediction'], 'ro', label='prediction')
plt.xticks(rotation='60')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Maximum Temperature (°C)')
plt.title('Actual and Predicted Maximum Temperatures')
plt.show()
