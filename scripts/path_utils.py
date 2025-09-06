"""
Path utilities for finding data files from different working directories
"""
import os

# Constants
STOCK_DATA_FILENAME = "VNI_2020_2025_FINAL.csv"

def get_stock_symbol():
    """
    Extract stock symbol from filename
    
    Returns:
        str: Stock symbol (e.g., 'VNI', 'FPT', 'CMG', 'ELC')
    """
    filename = STOCK_DATA_FILENAME
    # Extract symbol before first underscore
    return filename.split('_')[0]

def get_combined_filename():
    """
    Generate combined market data filename based on current stock
    
    Returns:
        str: Combined filename (e.g., 'VNI_Global_Markets_Combined_2020_2025.csv')
    """
    symbol = get_stock_symbol()
    return f"{symbol}_Global_Markets_Combined_2020_2025.csv"

def find_stock_data_path():
    """
    Find the correct path to stock csv file regardless of working directory
    
    Returns:
        str: Correct path to the stock data file
    """
    # Possible paths depending on where script is run from
    paths_to_try = [
        f'../data/{STOCK_DATA_FILENAME}',  # From scripts/ directory
        f'data/{STOCK_DATA_FILENAME}',     # From main directory
        f'/workspaces/stock_prediction_dashboard/data/{STOCK_DATA_FILENAME}'  # Absolute path
    ]
    
    for path in paths_to_try:
        if os.path.exists(path):
            return path
    
    raise FileNotFoundError(f"Could not find {STOCK_DATA_FILENAME} in any expected location")

def find_data_output_path(filename=None):
    """
    Find the correct output path for data files
    
    Args:
        filename (str): Name of the output file. If None, uses combined filename
        
    Returns:
        str: Correct output path
    """
    if filename is None:
        filename = get_combined_filename()
        
    # Check if we're in scripts/ directory
    if os.path.exists('../data/'):
        return f'../data/{filename}'
    # Check if we're in main directory  
    elif os.path.exists('data/'):
        return f'data/{filename}'
    else:
        return filename  # Fallback to current directory

# Backward compatibility alias
def find_vni_data_path():
    """
    Backward compatibility alias for find_stock_data_path()
    
    Returns:
        str: Correct path to the stock data file
    """
    return find_stock_data_path()
