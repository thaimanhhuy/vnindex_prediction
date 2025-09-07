# VNIndex Stock Prediction Dashboard

## Project Description

VNIndex Stock Prediction Dashboard is an interactive web application built with Streamlit that helps users forecast the VNIndex price based on historical data and technical indicators. The app supports both deep learning models (LSTM/GRU) and traditional ARIMA models, allowing users to compare prediction performance between methods.

## Key Features

- **Data Analysis**: Visualize candlestick charts, volume, RSI, MACD, MA20, MA50, and detailed data tables.
- **Model Configuration**: Customize LSTM/GRU model parameters (number of layers, neurons, dropout, batch size, etc.).
- **Model Training**: Preprocess data, train, and evaluate models with MAE, MSE, RMSE, and R² metrics.
- **Future Prediction**: Predict VNIndex prices for multiple future days, with results shown in tables and charts.
- **Model Comparison**: Compare the performance of LSTM/GRU and ARIMA models.
- **Model Management**: Save, load, and delete trained models.
- **Auto Data Crawler**: `scripts/vnindex_crawler_and_merge.py` automatically fetches daily VNIndex data from CafeF and merges it into `data/VNI_2020_2025_FINAL.csv`.

## Data Requirements

- The CSV file must contain the following columns: `Date`, `Open`, `High`, `Low`, `Close`, `Volume`.

## Usage Guide

1. **(Recommended) Create a virtual environment**:

   ```bash
   python3 -m venv .venv
   . .venv/bin/activate
   ```
2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```
3. **Run the app**:

   ```bash
   # Standard
   streamlit run app.py

   # If running inside a dev container/remote environment (optional)
   streamlit run app.py --server.address 0.0.0.0 --server.port 8501
   ```

4. **Using the dashboard**:

   - Explore and analyze historical data.
   - Configure and train models.
   - Predict future prices and compare models.
   - Save and manage trained models.

### Auto data crawler (CafeF)

Use the provided script to fetch the latest VNIndex data from CafeF and append it to your CSV:

```bash
python scripts/vnindex_crawler_and_merge.py
```

Notes:

- Requires Google Chrome and ChromeDriver (the script uses webdriver-manager to auto-install the right driver).
- Output file: `data/VNI_2020_2025_FINAL.csv` (new rows will be appended and sorted by date).
- Configs: adjust `NUM_DAYS`, `URL`, and `CSV_PATH` in `scripts/vnindex_crawler_and_merge.py` if needed.

## Main Files & Structure

- `app.py`: Main Streamlit application source code.
- `arima_model.py`, `predict_future.py`: Functions for ARIMA modeling and future prediction.
- `scripts/vnindex_crawler_and_merge.py`: Daily data crawler from CafeF and CSV merger.
- `data/VNI_2020_2025_FINAL.csv`: default data.
- `requirements.txt`: List of required Python packages.

## Technologies Used

- Python, Streamlit, TensorFlow/Keras, scikit-learn, Plotly, pandas, numpy, ta

## Contribution & Contact

- For contributions, bug reports, or suggestions: Please open an issue or pull request on this repository.
- Contact: [huythaimanh@gmail.com]
