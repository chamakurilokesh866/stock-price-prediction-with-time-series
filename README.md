# Stock Price Prediction with Time Series (LSTM)

This project uses Long Short-Term Memory (LSTM) networks to predict the stock price of MRF Limited based on historical data.

## Project Structure
- `main.py`: The core script that performs data loading, preprocessing, model training, and prediction.
- `MRF.NS.csv`: The dataset containing historical stock data.
- `requirements.txt`: List of Python dependencies.
- `automate.ps1`: A PowerShell script to automate installation, execution, and pushing to GitHub.
- `Stock Market Prediction using LSTM.ipynb`: The original research notebook.

## How to Run

### Automatic Way (Windows)
Right-click `automate.ps1` and select **Run with PowerShell**, or run it from your terminal:
```powershell
./automate.ps1
```

### Manual Way
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the script:**
   ```bash
   python main.py
   ```
3. **Check results:**
   - `stock_price_history.png`: Visualization of historical data.
   - `prediction_results.png`: Visualization of the model's performance.

## GitHub Integration
To push this project to your GitHub:
1. Create a new repository on GitHub.
2. Run `automate.ps1` and provide the repository URL when prompted.
