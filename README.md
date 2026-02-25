# ai-fraud-detection-system
AI-powered fraud detection desktop app built with Python &amp; Tkinter. Uses Random Forest &amp; Logistic Regression to detect suspicious transactions in real-time. Features OTP verification, email alerts, SQLite database, anomaly detection, live charts, and model save/load. Supports .env for secure credentials.
# 💳 AI-Based Fraud Detection System

An AI-powered desktop application for real-time fraud detection built with Python. Uses Machine Learning models to analyze transactions and flag suspicious activity with OTP verification and email alerts.

---

## 🖥️ Screenshots

> Train the model, predict transactions, and view live performance charts — all in one app.

---

## ✨ Features

- 🤖 **Machine Learning** — Random Forest & Logistic Regression models
- 🔍 **Real-Time Prediction** — Instant fraud probability scoring
- 📍 **Anomaly Detection** — Location, time, and amount anomaly checks
- 🔐 **OTP Verification** — 6-digit OTP with 5-minute expiry
- 📧 **Email Alerts** — Gmail integration for fraud notifications
- 🗄️ **SQLite Database** — Stores users, transactions, and alerts
- 📊 **Live Charts** — ROC curves, confusion matrix, feature importance
- 💾 **Model Persistence** — Auto-saves and loads trained models
- 📤 **Export** — Export data as CSV and reports as TXT
- 🔒 **Secure Credentials** — Gmail credentials stored in `.env` file

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/silent-jasoos/ai-fraud-detection-system.git
cd ai-fraud-detection-system
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup Email Credentials (Optional)
Rename `.env.example` to `.env` and fill in your Gmail credentials:
```
FRAUD_SENDER_EMAIL=your_email@gmail.com
FRAUD_SENDER_PASS=your_gmail_app_password
```

> **How to get Gmail App Password:**
> Google Account → Security → 2-Step Verification → App Passwords

### 4. Run the App
```bash
python fraud_detection.py
```

---

## 📁 Project Structure

```
ai-fraud-detection-system/
│
├── fraud_detection.py      # Main application file
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variables template
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

---

## 🧠 How It Works

1. **Train** — Generates synthetic transaction data and trains ML models
2. **Predict** — Enter transaction details to get a fraud risk score (0–100%)
3. **Anomaly Boost** — Location, time, and amount anomalies increase risk score
4. **OTP Alert** — Suspicious transactions trigger OTP verification
5. **Email Alert** — Fraud alerts sent to user's registered email
6. **Feedback** — User approves or denies transaction, improving the model

---

## 📊 Model Performance

| Metric | Random Forest | Logistic Regression |
|--------|--------------|---------------------|
| ROC-AUC | ~0.97+ | ~0.95+ |
| Recall | ~0.85+ | ~0.80+ |
| Accuracy | ~0.99+ | ~0.98+ |

> Results may vary depending on training settings.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.x | Core language |
| Tkinter | Desktop GUI |
| Scikit-learn | ML models |
| Pandas & NumPy | Data processing |
| Matplotlib | Charts & graphs |
| SQLite3 | Database |
| Joblib | Model save/load |
| SMTP / Gmail | Email alerts |

---

## ⚙️ Requirements

```
numpy
pandas
matplotlib
scikit-learn
joblib
```

Install all with:
```bash
pip install -r requirements.txt
```

---

## 🔒 Security Notes

- Never commit your `.env` file — it's in `.gitignore`
- Use Gmail **App Passwords** only, never your real Gmail password
- The `.env.example` file is a safe template with no real credentials

---

## 👤 Demo Users

| User ID | Name | Location | Avg Amount |
|---------|------|----------|------------|
| USR001 | Ali Hassan | Karachi | $1,500 |
| USR002 | Sara Ahmed | Lahore | $800 |
| USR003 | Zeeshan Khan | Islamabad | $3,000 |
| USR004 | Fatima Malik | Karachi | $500 |

---

## 📄 License

This project is open source and available under the [MIT License](https://github.com/silent-jasoos/ai-fraud-detection-system/blob/main/LICENSE).

---

## 🙌 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

> Built with ❤️ using Python & Machine Learning
