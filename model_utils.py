# Only keep model-related imports and functions
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# Data preprocessing for LSTM/GRU
def create_dataset(dataset, time_step, target_col_index):
    X, Y = [], []
    for i in range(len(dataset) - time_step):
        # Input X: Dữ liệu từ ngày t, t+1, ..., t+time_step-1  
        X.append(dataset[i:(i + time_step), :])
        # Target Y: Giá trị của ngày t+time_step
        Y.append(dataset[i + time_step, target_col_index])
    
    return np.array(X), np.array(Y)

def preprocess_data(
    df, 
    features_to_use, 
    target_column, 
    time_step=50, 
    scaler_type='minmax', 
    train_ratio=0.8
):
    """
    Chuẩn hóa và tạo tập train/test cho LSTM/GRU.
    QUAN TRỌNG: Chỉ fit scaler trên tập train để tránh data leakage.
    """
    if target_column not in features_to_use:
        raise ValueError(f"Target column '{target_column}' phải có trong danh sách features.")
    if df[features_to_use].isnull().any().any():
        raise ValueError("Dữ liệu có giá trị thiếu. Hãy xử lý trước khi huấn luyện.")

    data_to_scale = df[features_to_use].values
    target_col_index = features_to_use.index(target_column)
    
    # Chia dữ liệu TRƯỚC KHI scaling để tránh data leakage
    train_size = int(len(data_to_scale) * train_ratio)
    train_data_raw = data_to_scale[:train_size, :]
    test_data_raw = data_to_scale[train_size:, :]

    # Chọn scaler và CHỈ fit trên tập train
    scaler = MinMaxScaler() if scaler_type == 'minmax' else StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data_raw)  # Fit và transform train
    test_data_scaled = scaler.transform(test_data_raw)       # Chỉ transform test
    
    # Ghép lại để có toàn bộ dữ liệu đã scaled (cho mục đích khác nếu cần)
    scaled_data = np.vstack([train_data_scaled, test_data_scaled])

    # Sử dụng hàm create_dataset độc lập
    X_train, y_train = create_dataset(train_data_scaled, time_step, target_col_index)
    X_test, y_test = create_dataset(test_data_scaled, time_step, target_col_index)
    
    return X_train, y_train, X_test, y_test, scaler, scaled_data

def build_model(
    model_type, 
    time_step, 
    num_features, 
    num_neurons, 
    dropout_rate, 
    num_hidden_layers=2, 
    loss_fn='mae',
    learning_rate=0.001,
    use_batch_norm=False
):
    """
    Xây dựng model LSTM/GRU cho regression.
    """
    print(f"Building {model_type} model | Layers: {num_hidden_layers} | Neurons: {num_neurons} | Dropout: {dropout_rate} | Loss: {loss_fn} | Batch Norm: {use_batch_norm}")
    if model_type not in ['LSTM', 'GRU']:
        raise ValueError("Model type must be either 'LSTM' or 'GRU'.")
    model = Sequential()
    model.add(Input(shape=(time_step, num_features)))
    for i in range(num_hidden_layers - 1):
        if model_type == 'LSTM':
            model.add(LSTM(num_neurons, return_sequences=True))
        else:
            model.add(GRU(num_neurons, return_sequences=True))
        if use_batch_norm:
            model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

    if model_type == 'LSTM':
        model.add(LSTM(num_neurons))
    else:
        model.add(GRU(num_neurons))
    if use_batch_norm:
        model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='linear'))  # Regression nên dùng linear
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name='mae'),
            tf.keras.metrics.MeanSquaredError(name='mse')
        ]
    )
    return model

def train_model(model, X_train, y_train, config, callbacks=None):
    """
    Huấn luyện mô hình với các tham số và callbacks được cung cấp.
    """
    history = model.fit(
        X_train, y_train,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        validation_split=config['validation_split'],
        callbacks=callbacks,
        verbose=0
    )
    return history

# Model evaluation

def evaluate_model(model, X_test, y_test, scaler, target_col_index):
    """
    Đánh giá mô hình và thực hiện inverse transform hiệu quả cho cột mục tiêu.
    Hỗ trợ cả MinMaxScaler và StandardScaler.
    """
    y_pred_scaled = model.predict(X_test).flatten()

    # Inverse transform đúng cho từng loại scaler
    if hasattr(scaler, 'min_'):
        # MinMaxScaler: X_scaled = (X - min) / (max - min)
        # => X = X_scaled * (max - min) + min
        data_min = scaler.data_min_[target_col_index]
        data_range = scaler.data_range_[target_col_index]
        
        y_pred_inv = y_pred_scaled * data_range + data_min
        y_test_inv = y_test * data_range + data_min
    else:
        # StandardScaler: X_scaled = (X - mean) / std
        # => X = X_scaled * std + mean
        mean = scaler.mean_[target_col_index]
        std = scaler.scale_[target_col_index]
        
        y_pred_inv = y_pred_scaled * std + mean
        y_test_inv = y_test * std + mean

    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    mse = mean_squared_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_inv, y_pred_inv)
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R²': r2
    }, y_test_inv, y_pred_inv
