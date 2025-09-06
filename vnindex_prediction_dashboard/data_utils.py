import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD

try:
    import holidays
except ModuleNotFoundError:
    holidays = None

# Data loading

def load_data(file_path):
    try:
        data = pd.read_csv(file_path, index_col='Date', parse_dates=True)
        data.sort_index(inplace=True)
        return data
    except Exception as e:
        raise RuntimeError(f"Lỗi khi tải dữ liệu: {e}")

# Technical indicators

def add_technical_indicators(df):
    df_copy = df.copy()
    df_copy['RSI'] = RSIIndicator(close=df_copy['Close'], window=14).rsi()
    macd = MACD(close=df_copy['Close'])
    df_copy['MACD'] = macd.macd()
    df_copy['MACD_Signal'] = macd.macd_signal()
    df_copy['MACD_Histogram'] = macd.macd_diff()
    df_copy['MA_20'] = df_copy['Close'].rolling(window=20).mean()
    df_copy['MA_50'] = df_copy['Close'].rolling(window=50).mean()
    return df_copy

def find_missing_dates(df):
    """
    Kiểm tra ngày bị thiếu trừ thứ 7, chủ nhật và ngày lễ Việt Nam.
    Trả về danh sách các ngày bị thiếu.
    """
    # Tạo danh sách tất cả ngày trong khoảng
    all_dates = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    # Lấy ngày lễ Việt Nam
    if holidays is None:
        raise RuntimeError("Package 'holidays' not found. Please install it with `pip install holidays`.")
    vn_holidays = holidays.Vietnam()
    # Lọc ra ngày làm việc (thứ 2-6), không phải ngày lễ và không có trong dữ liệu
    missing = [d for d in all_dates if d.weekday() < 5 and d not in vn_holidays and d not in df.index]
    return missing

def count_missing_by_month(df):
    """
    Trả về dict với số ngày missing theo từng tháng (format 'YYYY-MM').
    """
    # Lấy danh sách ngày missing
    missing = find_missing_dates(df)
    if not missing:
        return {}
    # Chuyển thành DataFrame để nhóm
    df_missing = pd.DataFrame({'date': missing})
    # Tạo cột tháng
    df_missing['month'] = df_missing['date'].dt.to_period('M')
    # Đếm theo tháng
    counts = df_missing.groupby('month').size()
    # Chuyển sang dict với key là string
    return {str(month): int(count) for month, count in counts.items()}
