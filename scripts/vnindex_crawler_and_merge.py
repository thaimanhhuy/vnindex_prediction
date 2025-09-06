from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from datetime import datetime
import pandas as pd
import time
import os

# Constants
NUM_DAYS = 10
MAX_RETRIES = 5
URL = 'https://cafef.vn/du-lieu/lich-su-giao-dich-symbol-vnindex/trang-1-0-tab-1.chn'
CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'VNI_2020_2025_FINAL.csv')

# Merge dữ liệu mới vào CSV
def merge_data(new_df, csv_path):
    try:
        existing_df = pd.read_csv(csv_path)
    except:
        existing_df = pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    
    existing_df['Date'] = pd.to_datetime(existing_df['Date'])
    new_df['Date'] = pd.to_datetime(new_df['Date'])
    
    if len(existing_df) > 0:
        last_date = existing_df['Date'].max()
        new_data = new_df[new_df['Date'] > last_date]
        if len(new_data) == 0:
            print("ℹ️ Không có dữ liệu mới")
            return
        print(f"📊 Thêm {len(new_data)} ngày mới")
        merged_df = pd.concat([existing_df, new_data], ignore_index=True)
    else:
        merged_df = new_df.copy()
    
    merged_df = merged_df.sort_values('Date').reset_index(drop=True)
    merged_df['Date'] = merged_df['Date'].dt.strftime('%Y-%m-%d')
    merged_df.to_csv(csv_path, index=False)
    print(f"✅ Lưu {len(merged_df)} dòng dữ liệu")

# Cấu hình Chrome
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--disable-gpu')
chrome_options.add_argument('--no-sandbox')

def crawl_data_from_web(driver):
    """Crawl dữ liệu từ CafeF"""
    # Hàm kiểm tra bảng đã load chưa
    def is_table_loaded(driver):
        try:
            rows = driver.find_elements(By.CSS_SELECTOR, '#render-table-owner tr')
            return len(rows) > 0
        except:
            return False
    
    # Tải lại trang cho đến khi bảng load thành công
    retry_count = 0

    while retry_count < MAX_RETRIES:
        try:
            driver.get(URL)
            print(f"📡 Đang tải trang... (lần thử {retry_count + 1})")
            time.sleep(3)
            if is_table_loaded(driver):
                print("✅ Trang web đã load thành công!")
                break
            retry_count += 1
        except Exception as e:
            print("❌ Lỗi khi tải trang:", e)
            retry_count += 1
            time.sleep(3)
    else:
        print(f"❌ Không thể tải dữ liệu sau {MAX_RETRIES} lần thử.")
        driver.quit()
        raise Exception("Không thể tải trang web")

    data = []
    try:
        table_rows = driver.find_elements(By.CSS_SELECTOR, '#render-table-owner tr')
        print(f"📊 Tìm thấy {len(table_rows)} dòng dữ liệu trong bảng")

        for i, row in enumerate(table_rows[:NUM_DAYS]):
            cols = row.find_elements(By.TAG_NAME, 'td')
            if len(cols) >= 11:
                date_str = cols[0].text.strip()       # Ngày
                close = cols[1].text.strip().replace(',', '')  # Đóng cửa
                open_price = cols[8].text.strip().replace(',', '')  # Mở cửa
                high = cols[9].text.strip().replace(',', '')  # Cao nhất
                low = cols[10].text.strip().replace(',', '')  # Thấp nhất
                volume = cols[4].text.strip().replace(',', '').replace('.', '')  # Khối lượng

                # Kiểm tra dữ liệu hợp lệ - bỏ qua nếu Open hoặc High = 0
                if open_price == '0' or high == '0':
                    print(f"⚠️ Bỏ qua ngày {date_str}: dữ liệu Open ({open_price}) hoặc High ({high}) = 0")
                    continue

                # Chuẩn hóa ngày
                formatted_date = datetime.strptime(date_str, "%d/%m/%Y").strftime("%Y-%m-%d")
                
                data.append([formatted_date, open_price, high, low, close, volume])
                print(f"✓ Lấy dữ liệu ngày {formatted_date}: Close = {close}")

        if not data:
            raise Exception("Không có dữ liệu để crawl")
            
        # Tạo DataFrame
        df_new = pd.DataFrame(data, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])

        # Chuyển đổi kiểu dữ liệu
        df_new['Open'] = pd.to_numeric(df_new['Open'], errors='coerce')
        df_new['High'] = pd.to_numeric(df_new['High'], errors='coerce')
        df_new['Low'] = pd.to_numeric(df_new['Low'], errors='coerce')
        df_new['Close'] = pd.to_numeric(df_new['Close'], errors='coerce')
        df_new['Volume'] = pd.to_numeric(df_new['Volume'], errors='coerce').fillna(0).astype(int)

        print(f"\n🎉 Crawl thành công {len(df_new)} ngày dữ liệu:")
        print(df_new.to_string(index=False))
        
        return df_new

    except Exception as e:
        print(f"❌ Lỗi khi trích xuất dữ liệu: {e}")
        raise e
    finally:
        # Đảm bảo driver luôn được đóng
        driver.quit()

# Khởi tạo trình duyệt và crawl dữ liệu
# Sử dụng webdriver-manager để tự động tải ChromeDriver phù hợp
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)
print("✅ ChromeDriver khởi tạo thành công với webdriver-manager")

# Crawl dữ liệu từ web
print("🌐 Bắt đầu crawl dữ liệu từ CafeF...")
df_new = crawl_data_from_web(driver)

# Merge dữ liệu từ crawl
print("\n🔄 Bắt đầu merge dữ liệu...")
merge_data(df_new, CSV_PATH)
print("\n🎉 Hoàn thành!")
