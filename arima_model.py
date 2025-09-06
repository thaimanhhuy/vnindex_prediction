import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def check_stationarity(timeseries):
    """
    Kiểm tra tính dừng của chuỗi thời gian bằng ADF test
    """
    result = adfuller(timeseries)
    return {
        'adf_statistic': result[0],
        'p_value': result[1],
        'critical_values': result[4],
        'is_stationary': result[1] <= 0.05
    }

def find_optimal_arima_params(data, seasonal=False):
    """
    Tìm tham số tối ưu cho mô hình ARIMA bằng cách thử nghiệm
    """
    best_aic = float('inf')
    best_order = None
    
    # Thử nghiệm các tham số p, d, q
    for p in range(0, 4):
        for d in range(0, 3):
            for q in range(0, 4):
                try:
                    model = ARIMA(data, order=(p, d, q))
                    fitted_model = model.fit()
                    if fitted_model.aic < best_aic:
                        best_aic = fitted_model.aic
                        best_order = (p, d, q)
                except:
                    continue
    
    return best_order if best_order else (1, 1, 1)

def train_arima_model(train_data, order=None):
    """
    Huấn luyện mô hình ARIMA
    """
    try:
        if order is None:
            # Tự động tìm tham số tối ưu
            order = find_optimal_arima_params(train_data)
        
        # Huấn luyện với tham số đã cho
        model = ARIMA(train_data, order=order)
        fitted_model = model.fit()
        return fitted_model, order
    except Exception as e:
        print(f"Lỗi khi huấn luyện ARIMA: {e}")
        # Fallback với tham số mặc định
        try:
            model = ARIMA(train_data, order=(1, 1, 1))
            fitted_model = model.fit()
            return fitted_model, (1, 1, 1)
        except:
            return None, None

def predict_arima(model, steps):
    """
    Dự đoán với mô hình ARIMA
    """
    try:
        forecast = model.forecast(steps=steps)
        return forecast
    except Exception as e:
        print(f"Lỗi khi dự đoán ARIMA: {e}")
        return None

def evaluate_arima_model(y_true, y_pred):
    """
    Đánh giá hiệu suất mô hình ARIMA
    """
    try:
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R²': r2
        }
    except Exception as e:
        print(f"Lỗi khi đánh giá ARIMA: {e}")
        return None

def prepare_data_for_arima(data, target_column='Close'):
    """
    Chuẩn bị dữ liệu cho mô hình ARIMA
    """
    # Chỉ lấy cột target
    series = data[target_column].dropna()
    
    # Kiểm tra tính dừng
    stationarity_test = check_stationarity(series)
    
    return series, stationarity_test

def compare_models_performance(lstm_metrics, arima_metrics):
    """
    So sánh hiệu suất giữa LSTM/GRU và ARIMA
    """
    comparison = {}
    
    for metric in ['MAE', 'MSE', 'RMSE', 'R²']:
        lstm_val = lstm_metrics.get(metric, 0)
        arima_val = arima_metrics.get(metric, 0)
        
        if metric == 'R²':
            # R² càng cao càng tốt
            better_model = 'LSTM/GRU' if lstm_val > arima_val else 'ARIMA'
            improvement = abs(lstm_val - arima_val)
        else:
            # MAE, MSE, RMSE càng thấp càng tốt
            better_model = 'LSTM/GRU' if lstm_val < arima_val else 'ARIMA'
            improvement = abs(lstm_val - arima_val) / max(lstm_val, arima_val) * 100
        
        comparison[metric] = {
            'LSTM/GRU': lstm_val,
            'ARIMA': arima_val,
            'better_model': better_model,
            'improvement_pct': improvement
        }
    
    return comparison

