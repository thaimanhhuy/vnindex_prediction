#!/usr/bin/env python3
"""
Script huấn luyện mô hình LSTM và dự đoán 20 ngày tiếp theo
Sử dụng thông số mặc định đã chỉ định
"""

import pandas as pd
import numpy as np
import pickle
import os
import json
from datetime import datetime, timedelta
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# Import các hàm cần thiết
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_utils import load_data, add_technical_indicators
from model_utils import preprocess_data, build_model, train_model, evaluate_model

# ===== CONSTANTS - Dễ dàng thay đổi =====
MODEL_TYPE = 'LSTM'  # Thay đổi thành 'GRU' nếu muốn sử dụng GRU

def main():
    print(f"🚀 Bắt đầu huấn luyện mô hình {MODEL_TYPE}...")
    
    # ===== THÔNG SỐ MÔ HÌNH =====
    MODEL_CONFIG = {
        'model_type': MODEL_TYPE,
        'num_neurons': 64,
        'dropout_rate': 0.35,
        'num_hidden_layers': 2,
        'learning_rate': 0.001,
        'loss_fn': 'mae',
        'use_batch_norm': False
    }
    
    # ===== THÔNG SỐ HUẤN LUYỆN =====
    TRAIN_CONFIG = {
        'epochs': 50,
        'batch_size': 32,
        'validation_split': 0.1,
        'time_step': 50
    }
    
    # ===== THÔNG SỐ DỮ LIỆU =====
    DATA_CONFIG = {
        'features_to_use': ['Close', 'Volume', 'RSI', 'MACD'],
        'target_column': 'Close',
        'scaler_type': 'minmax',
        'train_ratio': 0.8
    }
    
    PREDICTION_DAYS = 20
    
    print(f"📊 Cấu hình mô hình: {MODEL_CONFIG}")
    print(f"📈 Cấu hình huấn luyện: {TRAIN_CONFIG}")
    print(f"📋 Cấu hình dữ liệu: {DATA_CONFIG}")
    
    # ===== TÀI DỮ LIỆU =====
    print("\n📂 Đang tải dữ liệu...")
    data = load_data('../data/VNI_2020_2025_FINAL.csv')
    print(f"✅ Đã tải {len(data)} dòng dữ liệu từ {data.index[0]} đến {data.index[-1]}")
    
    # ===== THÊM CHỈ BÁO KỸ THUẬT =====
    print("\n📈 Đang tính toán các chỉ báo kỹ thuật...")
    data_with_indicators = add_technical_indicators(data)
    
    # Kiểm tra dữ liệu sau khi thêm indicators
    print(f"✅ Đã thêm chỉ báo kỹ thuật. Dữ liệu hiện có {len(data_with_indicators)} dòng")
    print(f"📋 Các cột: {list(data_with_indicators.columns)}")
    
    # Loại bỏ dữ liệu thiếu
    data_clean = data_with_indicators.dropna()
    print(f"🧹 Sau khi loại bỏ dữ liệu thiếu: {len(data_clean)} dòng")
    
    # ===== CHUẨN BỊ DỮ LIỆU =====
    print("\n🔄 Đang chuẩn bị dữ liệu cho huấn luyện...")
    X_train, y_train, X_test, y_test, scaler, scaled_data = preprocess_data(
        df=data_clean,
        features_to_use=DATA_CONFIG['features_to_use'],
        target_column=DATA_CONFIG['target_column'],
        time_step=TRAIN_CONFIG['time_step'],
        scaler_type=DATA_CONFIG['scaler_type'],
        train_ratio=DATA_CONFIG['train_ratio']
    )
    
    print(f"✅ Dữ liệu đã được chuẩn bị:")
    print(f"   📊 X_train shape: {X_train.shape}")
    print(f"   📊 y_train shape: {y_train.shape}")
    print(f"   📊 X_test shape: {X_test.shape}")
    print(f"   📊 y_test shape: {y_test.shape}")
    
    # ===== XÂY DỰNG MÔ HÌNH =====
    print("\n🏗️ Đang xây dựng mô hình...")
    model = build_model(
        model_type=MODEL_CONFIG['model_type'],
        time_step=TRAIN_CONFIG['time_step'],
        num_features=len(DATA_CONFIG['features_to_use']),
        num_neurons=MODEL_CONFIG['num_neurons'],
        dropout_rate=MODEL_CONFIG['dropout_rate'],
        num_hidden_layers=MODEL_CONFIG['num_hidden_layers'],
        loss_fn=MODEL_CONFIG['loss_fn'],
        learning_rate=MODEL_CONFIG['learning_rate'],
        use_batch_norm=MODEL_CONFIG['use_batch_norm']
    )
    
    print("✅ Mô hình đã được xây dựng thành công!")
    model.summary()
    
    # ===== HUẤN LUYỆN MÔ HÌNH =====
    print("\n🎯 Bắt đầu huấn luyện mô hình...")
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    history = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        config=TRAIN_CONFIG,
        callbacks=callbacks
    )
    
    print("✅ Huấn luyện hoàn thành!")
    
    # ===== ĐÁNH GIÁ MÔ HÌNH =====
    print("\n📊 Đang đánh giá mô hình...")
    target_col_index = DATA_CONFIG['features_to_use'].index(DATA_CONFIG['target_column'])
    
    metrics, y_test_inv, y_pred_inv = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        scaler=scaler,
        target_col_index=target_col_index
    )
    
    print("📈 Kết quả đánh giá mô hình:")
    for metric, value in metrics.items():
        print(f"   {metric}: {value:.4f}")
    
    # ===== LƯU MÔ HÌNH =====
    print("\n💾 Đang lưu mô hình...")
    
    # Tạo thư mục models nếu chưa có
    os.makedirs('../models', exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_type_name = MODEL_CONFIG['model_type'].lower()  # lstm hoặc gru
    
    # Lưu model với tên bao gồm loại mô hình
    model_path = f'../models/model_{model_type_name}_{timestamp}.h5'
    model.save(model_path)
    print(f"✅ Đã lưu mô hình: {model_path}")
    
    # Lưu scaler
    scaler_path = f'../models/model_{model_type_name}_{timestamp}_scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✅ Đã lưu scaler: {scaler_path}")
    
    # Lưu config
    config_path = f'../models/model_{model_type_name}_{timestamp}_config.pkl'
    config = {
        'model_config': MODEL_CONFIG,
        'train_config': TRAIN_CONFIG,
        'data_config': DATA_CONFIG,
        'metrics': metrics,
        'features_to_use': DATA_CONFIG['features_to_use'],
        'target_column': DATA_CONFIG['target_column']
    }
    with open(config_path, 'wb') as f:
        pickle.dump(config, f)
    print(f"✅ Đã lưu cấu hình: {config_path}")
    
    # ===== DỰ ĐOÁN 20 NGÀY TIẾP THEO =====
    print(f"\n🔮 Bắt đầu dự đoán {PREDICTION_DAYS} ngày tiếp theo...")
    
    # Import hàm dự đoán chuẩn
    from predict_future import predict_future
    
    # Sử dụng hàm predict_future chuẩn
    predictions = predict_future(
        model=model,
        data=scaled_data,
        time_step=TRAIN_CONFIG['time_step'],
        n_future=PREDICTION_DAYS,
        scaler=scaler,
        num_features=len(DATA_CONFIG['features_to_use']),
        target_col_index=target_col_index,
        features_to_use=DATA_CONFIG['features_to_use']
    )
    
    # In kết quả dự đoán
    for i, pred in enumerate(predictions):
        print(f"   Ngày {i+1}: {pred:.2f}")
    
    # ===== TẠO DATAFRAME DỰ ĐOÁN =====
    # Tạo ngày cho dự đoán (chỉ ngày giao dịch)
    last_date = data_clean.index[-1]
    future_dates = []
    days_added = 0
    current_date = last_date
    
    while days_added < PREDICTION_DAYS:
        current_date += timedelta(days=1)
        if current_date.weekday() < 5:  # 0=Mon, ..., 4=Fri
            future_dates.append(current_date)
            days_added += 1
    
    # Tạo DataFrame kết quả
    prediction_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Close': predictions
    })
    prediction_df.set_index('Date', inplace=True)
    
    print(f"\n🎯 Dự đoán {PREDICTION_DAYS} ngày tiếp theo:")
    print(prediction_df.round(2))
    
    # Lưu kết quả dự đoán
    prediction_path = f'../models/predictions_{model_type_name}_{timestamp}.csv'
    prediction_df.to_csv(prediction_path)
    print(f"\n💾 Đã lưu kết quả dự đoán: {prediction_path}")
    
    print("\n🎉 Hoàn thành! Mô hình đã được huấn luyện và dự đoán thành công!")
    
    return {
        'model': model,
        'scaler': scaler,
        'metrics': metrics,
        'predictions': prediction_df,
        'model_path': model_path,
        'config': config
    }

if __name__ == "__main__":
    results = main()
