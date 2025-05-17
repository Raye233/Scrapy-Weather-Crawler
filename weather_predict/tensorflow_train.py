import os
import tensorflow as tf
import joblib
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from keras.layers import Dropout

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False # 解决坐标轴负号显示问题

def train():
    city = 'wuhan'

    df = pd.read_csv(r'../wuhan_year5_weather_data.csv', encoding='gbk')
    # df = pd.read_csv(r'../combined_file.csv', encoding='utf-8')

    # 处理时间特征
    df['时间'] = pd.to_datetime(df['时间'])
    df['年'] = df['时间'].dt.year
    df['月'] = df['时间'].dt.month
    df['日'] = df['时间'].dt.day

    # 添加季节特征（示例划分，可根据实际情况调整）
    seasons = {1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2,
               7: 2, 8: 2, 9: 3, 10: 3, 11: 3, 12: 0}
    df['季节'] = df['月'].map(seasons)

    # 2. 特征工程
    X = df[['年', '月', '日', '季节']]
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    # 3. 目标变量处理
    # 初始化编码器
    wind_dir_encoder = LabelEncoder()
    uv_encoder = LabelEncoder()

    # 编码分类变量
    df['风向'] = wind_dir_encoder.fit_transform(df['风向'])
    uv_labels = ['很强', '强', '中等', '弱', '最弱']  # 根据实际数据顺序调整
    uv_encoder.fit(uv_labels)
    df['紫外线'] = uv_encoder.transform(df['紫外线'])

    # 目标变量
    y = df[['最低温度', '最高温度', '湿度', '风向', '风力', '紫外线', '空气质量']]

    # 4. 数据标准化（数值型目标）
    numerical_cols = ['最低温度', '最高温度', '湿度', '空气质量']
    scaler_y = StandardScaler()
    y[numerical_cols] = scaler_y.fit_transform(y[numerical_cols])

    # 5. 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42)

    # 6. 构建DNN模型
    input_layer = Input(shape=(X_train.shape[1],))
    x = Dense(128, activation='relu')(input_layer)
    # x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    # x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)

    # 多任务输出层
    outputs = [
        Dense(1, name='min_temp')(x),  # 最低温度
        Dense(1, name='max_temp')(x),  # 最高温度
        Dense(1, name='humidity')(x),  # 湿度
        Dense(8, activation='softmax', name='wind_dir')(x),  # 风向
        Dense(7, activation='softmax', name='wind_force')(x),  # 风力
        Dense(5, activation='softmax', name='uv')(x),  # 紫外线
        Dense(1, name='air_quality')(x)  # 空气质量
    ]

    model = Model(inputs=input_layer, outputs=outputs)

    tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)

    # 7. 编译模型
    losses = {
        'min_temp': 'mse',
        'max_temp': 'mse',
        'humidity': 'mse',
        'wind_dir': 'sparse_categorical_crossentropy',
        'wind_force': 'sparse_categorical_crossentropy',
        'uv': 'sparse_categorical_crossentropy',
        'air_quality': 'mse'
    }

    model.compile(optimizer=Adam(0.001),
                  loss=losses,
                  metrics={
                      'wind_dir': 'accuracy',
                      'wind_force': 'accuracy',
                      'uv': 'accuracy'
                  })
    tf_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
    # 8. 训练模型
    history = model.fit(
        X_train,
        {
            'min_temp': y_train['最低温度'],
            'max_temp': y_train['最高温度'],
            'humidity': y_train['湿度'],
            'wind_dir': y_train['风向'],
            'wind_force': y_train['风力'],
            'uv': y_train['紫外线'],
            'air_quality': y_train['空气质量']
        },
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        verbose=1,
        callbacks=[tf_callback]
        )

    # 定义绘图函数
    def plot_training_history(history):
        plt.figure(figsize=(12, 8))

        # 绘制损失曲线
        plt.subplot(2, 2, 1)
        plt.plot(history.history['loss'], label='训练损失')
        plt.plot(history.history['val_loss'], label='验证损失')
        plt.title('损失曲线')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # 绘制准确率曲线
        plt.subplot(2, 2, 2)
        plt.plot(history.history['wind_dir_accuracy'], label='风向训练准确率')
        plt.plot(history.history['val_wind_dir_accuracy'], label='风向验证准确率')
        plt.plot(history.history['wind_force_accuracy'], label='风力训练准确率')
        plt.plot(history.history['val_wind_force_accuracy'], label='风力验证准确率')
        plt.plot(history.history['uv_accuracy'], label='紫外线训练准确率')
        plt.plot(history.history['val_uv_accuracy'], label='紫外线验证准确率')
        plt.title('分类准确率曲线')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()

    # 调用绘图函数
    plot_training_history(history)

    # 9. 预测示例
    def predict_weather(date_str):
        # 创建测试数据
        date_df = pd.DataFrame({'时间': [date_str]})
        date_df['时间'] = pd.to_datetime(date_df['时间'])
        date_df['年'] = date_df['时间'].dt.year
        date_df['月'] = date_df['时间'].dt.month
        date_df['日'] = date_df['时间'].dt.day
        date_df['季节'] = date_df['月'].map(seasons)

        X_new = date_df[['年', '月', '日', '季节']]
        X_new_scaled = scaler_X.transform(X_new)

        # 进行预测
        predictions = model.predict(X_new_scaled)

        # 处理预测结果
        # 数值型结果反标准化
        numerical = np.array([[predictions[0][0][0], predictions[1][0][0],
                               predictions[2][0][0], predictions[6][0][0]]])
        numerical = scaler_y.inverse_transform(numerical)

        # 分类结果解码
        wind_dir = wind_dir_encoder.inverse_transform([np.argmax(predictions[3][0])])[0]
        wind_force = np.argmax(predictions[4][0])
        uv = uv_encoder.inverse_transform([np.argmax(predictions[5][0])])[0]

        return {
            '最低温度': numerical[0][0],
            '最高温度': numerical[0][1],
            '湿度': numerical[0][2],
            '风向': wind_dir,
            '风力': int(wind_force),
            '紫外线': uv,
            '空气质量': numerical[0][3]
        }

    # 使用示例
    prediction = predict_weather('2025-05-07')
    for k, v in prediction.items():
        print(f'{k}: {v}')

    model.save(f'../tensorflow_model/{city}/weather_model.h5')  # HDF5格式

    # 保存预处理相关对象
    joblib.dump(scaler_X, f'../tensorflow_model/{city}/scaler_X.save')
    joblib.dump(scaler_y, f'../tensorflow_model/{city}/scaler_y.save')
    joblib.dump(wind_dir_encoder, f'../tensorflow_model/{city}/wind_dir_encoder.save')
    joblib.dump(uv_encoder, f'../tensorflow_model/{city}/uv_encoder.save')

    print("模型和预处理组件已保存")
    # --------------------------------------------------


if '__main__' == __name__:
    train()