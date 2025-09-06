import numpy as np
import pandas as pd
from collections import deque

def predict_future(model, data, time_step, n_future, scaler, num_features, target_col_index, features_to_use=None):
    """
    Dự đoán giá trị tương lai cho n_future bước tiếp theo.
    [REFACTOR] Phiên bản này giữ nguyên các feature khác để tránh sai số tích lũy.
    [OPTIMIZED] Sử dụng deque cho sliding window efficiency.
    
    Args:
        model: Trained model
        data: Scaled data array
        time_step: Number of time steps to use for prediction
        n_future: Number of future steps to predict
        scaler: Fitted scaler object
        num_features: Number of features
        target_col_index: Index of target column
        features_to_use: List of feature names (optional, for compatibility)
    """
    # Lấy chuỗi dữ liệu cuối cùng
    last_sequence = data[-time_step:]
    future_preds_scaled = []

    # [REFACTOR] Lấy các feature từ bước thời gian cuối cùng và giữ nguyên
    last_known_features = last_sequence[-1, :].copy()

    # [OPTIMIZED] Sử dụng deque cho sliding window thay vì np.append
    # Initialize deque with the last sequence for efficient sliding window
    sequence_deque = deque(last_sequence, maxlen=time_step)

    for step in range(n_future):
        # Convert deque to numpy array for model prediction
        current_sequence = np.array(sequence_deque).reshape(1, time_step, num_features)
        
        # Dự đoán bước tiếp theo
        next_pred_scaled = model.predict(current_sequence, verbose=0)[0][0]
        future_preds_scaled.append(next_pred_scaled)

        # [REFACTOR] Tạo vector feature mới
        # Bắt đầu với các feature đã biết cuối cùng
        new_feature_vector = last_known_features.copy()
        # Chỉ cập nhật giá trị dự đoán mới vào cột mục tiêu
        new_feature_vector[target_col_index] = next_pred_scaled
        
        # [REFACTOR] Không cần update_technical_indicators nữa
        # Giữ nguyên các feature khác để tránh sai số tích lũy

        # [OPTIMIZED] Cập nhật sliding window bằng deque - O(1) operation
        # Automatically removes oldest element when maxlen is reached
        sequence_deque.append(new_feature_vector)

    # Inverse transform các dự đoán
    future_preds_scaled = np.array(future_preds_scaled).reshape(-1, 1)

    # Tạo dummy array để inverse transform
    dummy_preds = np.zeros((len(future_preds_scaled), num_features))
    dummy_preds[:, target_col_index] = future_preds_scaled.flatten()
    future_preds_inv = scaler.inverse_transform(dummy_preds)[:, target_col_index]

    # In kết quả cuối cùng với format đẹp
    formatted_preds = [f"{pred:.2f}" for pred in future_preds_inv]
    print(f"🔮 Dự đoán {n_future} ngày tiếp theo: [{', '.join(formatted_preds)}]")
    
    return future_preds_inv