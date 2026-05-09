# Stock Price Prediction with Time Series (LSTM)

An end-to-end AI web application that uses **Long Short-Term Memory (LSTM)** networks to predict the stock price of **MRF Limited** (NSE: MRF.NS) based on 5 years of historical data.

---

## 🚀 Live Demo

Run it locally and open **http://localhost:5000** to see the interactive dashboard.

## 📸 Features

- 📈 **Interactive Charts** — Full history, test vs prediction overlay, error analysis
- 🤖 **LSTM Model** — 128 + 64 unit deep network trained on 5 years of OHLCV data
- 📊 **Stats Dashboard** — Live close price, AI prediction, high/low, volume
- 🗓️ **OHLCV Table** — Last 30 trading days with daily change indicators
- ⚡ **One-click Training** — Train the model directly from the browser UI

---

## 🗂️ Project Structure

```
├── app.py                          # Flask API backend
├── main.py                         # Standalone prediction script
├── static/
│   └── index.html                  # Interactive web dashboard (Chart.js)
├── MRF.NS.csv                      # Historical stock dataset (2018–2023)
├── Stock Market Prediction using LSTM.ipynb   # Research notebook
├── requirements.txt                # Python dependencies
└── automate.ps1                    # Windows automation script
```

---

## ⚙️ How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the web server
```bash
python app.py
```

### 3. Open your browser
```
http://localhost:5000
```

Click **"Train & Predict"** on the dashboard to run the LSTM model and view results.

---

## 🧠 Model Architecture

| Layer        | Units | Details               |
|-------------|-------|-----------------------|
| LSTM        | 128   | return_sequences=True |
| Dropout     | 20%   |                       |
| LSTM        | 64    | return_sequences=False |
| Dropout     | 20%   |                       |
| Dense       | 32    |                       |
| Dense (out) | 1     | Close price output    |

- **Optimizer**: Adam
- **Loss**: Mean Squared Error
- **Sequence length**: 60 days
- **Train/Test split**: 80% / 20%

---

## 📦 API Endpoints

| Method | Endpoint       | Description                        |
|--------|---------------|------------------------------------|
| GET    | `/`           | Serve the web dashboard            |
| GET    | `/api/status` | Check if model is trained          |
| POST   | `/api/train`  | Train the LSTM model               |
| GET    | `/api/data`   | Get predictions and chart data     |
| GET    | `/api/history`| Get last 30 days of OHLCV data     |

---

## 🛠️ Tech Stack

- **ML**: TensorFlow / Keras (LSTM)
- **Backend**: Flask + Flask-CORS
- **Frontend**: HTML, CSS, JavaScript + Chart.js
- **Data**: pandas, NumPy, scikit-learn
