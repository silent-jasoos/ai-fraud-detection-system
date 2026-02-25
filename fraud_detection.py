import sys
sys.stdout.reconfigure(encoding='utf-8')

# ─────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import threading
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, recall_score, accuracy_score
)
import sqlite3
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import random
import string
import datetime
import json
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# LOAD ENV CREDENTIALS SAFELY
# ─────────────────────────────────────────────
def load_env(path=".env"):
    """Load key=value pairs from a .env file into os.environ."""
    if os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())

load_env()

SENDER_EMAIL = os.environ.get("FRAUD_SENDER_EMAIL", "")
SENDER_PASS  = os.environ.get("FRAUD_SENDER_PASS",  "")

# ─────────────────────────────────────────────
# COLORS
# ─────────────────────────────────────────────
BG_DARK  = "#0d1117"
BG_CARD  = "#161b22"
BG_INPUT = "#21262d"
ACCENT   = "#238636"
ACCENT2  = "#1f6feb"
RED      = "#f85149"
GREEN    = "#3fb950"
YELLOW   = "#d29922"
PURPLE   = "#8957e5"
TEXT     = "#e6edf3"
TEXT_DIM = "#8b949e"
BORDER   = "#30363d"

# ─────────────────────────────────────────────
# GLOBAL MODEL VARIABLES
# ─────────────────────────────────────────────
model_rf   = None
model_lr   = None
scaler_amt = None
scaler_tim = None
X_test_g   = None
y_test_g   = None
df_global  = None
trained    = False
otp_store  = {}   # {user_id: otp_code}

MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# MODEL PERSISTENCE
# ─────────────────────────────────────────────
def save_models():
    """Save trained models and scalers to disk."""
    global model_rf, model_lr, scaler_amt, scaler_tim
    try:
        if model_rf:
            joblib.dump(model_rf,   os.path.join(MODEL_DIR, "model_rf.pkl"))
        if model_lr:
            joblib.dump(model_lr,   os.path.join(MODEL_DIR, "model_lr.pkl"))
        if scaler_amt:
            joblib.dump(scaler_amt, os.path.join(MODEL_DIR, "scaler_amt.pkl"))
        if scaler_tim:
            joblib.dump(scaler_tim, os.path.join(MODEL_DIR, "scaler_tim.pkl"))
        return True
    except Exception as e:
        print(f"Model save error: {e}")
        return False

def load_models():
    """Load previously trained models from disk if they exist."""
    global model_rf, model_lr, scaler_amt, scaler_tim, trained
    try:
        rf_path  = os.path.join(MODEL_DIR, "model_rf.pkl")
        lr_path  = os.path.join(MODEL_DIR, "model_lr.pkl")
        amt_path = os.path.join(MODEL_DIR, "scaler_amt.pkl")
        tim_path = os.path.join(MODEL_DIR, "scaler_tim.pkl")
        if os.path.exists(rf_path):
            model_rf   = joblib.load(rf_path)
        if os.path.exists(lr_path):
            model_lr   = joblib.load(lr_path)
        if os.path.exists(amt_path):
            scaler_amt = joblib.load(amt_path)
        if os.path.exists(tim_path):
            scaler_tim = joblib.load(tim_path)
        if model_rf or model_lr:
            trained = True
            return True
    except Exception as e:
        print(f"Model load error: {e}")
    return False

# ─────────────────────────────────────────────
# DATABASE SETUP
# ─────────────────────────────────────────────
DB_PATH = "fraud_system.db"

def init_database():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS users (
        user_id TEXT PRIMARY KEY,
        name TEXT,
        email TEXT,
        avg_amount REAL DEFAULT 0,
        usual_location TEXT DEFAULT 'Unknown',
        usual_hour_start INTEGER DEFAULT 9,
        usual_hour_end INTEGER DEFAULT 21,
        total_transactions INTEGER DEFAULT 0,
        fraud_count INTEGER DEFAULT 0,
        created_at TEXT
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS transactions (
        txn_id TEXT PRIMARY KEY,
        user_id TEXT,
        amount REAL,
        location TEXT,
        hour INTEGER,
        fraud_probability REAL,
        is_fraud INTEGER,
        confirmed_by_user INTEGER DEFAULT -1,
        timestamp TEXT,
        notes TEXT
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS alerts (
        alert_id TEXT PRIMARY KEY,
        user_id TEXT,
        txn_id TEXT,
        alert_type TEXT,
        otp_sent TEXT,
        otp_verified INTEGER DEFAULT 0,
        timestamp TEXT
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS model_log (
        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
        roc_auc REAL,
        accuracy REAL,
        recall REAL,
        total_trained INTEGER,
        timestamp TEXT
    )''')

    demo_users = [
        ("USR001", "Ali Hassan",   "ali@example.com",   1500, "Karachi",   9, 20, 45, 2),
        ("USR002", "Sara Ahmed",   "sara@example.com",   800, "Lahore",   10, 22, 30, 0),
        ("USR003", "Zeeshan Khan", "zeeshan@example.com",3000, "Islamabad", 8, 18, 60, 5),
        ("USR004", "Fatima Malik", "fatima@example.com",  500, "Karachi",  12, 21, 20, 1),
    ]
    for u in demo_users:
        c.execute('''INSERT OR IGNORE INTO users VALUES (?,?,?,?,?,?,?,?,?,?)''',
                  (*u, datetime.datetime.now().isoformat()))

    conn.commit()
    conn.close()
    print("Database initialized!")

def get_user(user_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE user_id=?", (user_id,))
    row = c.fetchone()
    conn.close()
    return row

def save_transaction(txn_id, user_id, amount, location, hour, prob, is_fraud):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''INSERT OR REPLACE INTO transactions VALUES (?,?,?,?,?,?,?,-1,?,?)''',
              (txn_id, user_id, amount, location, hour,
               prob, is_fraud, datetime.datetime.now().isoformat(), ""))
    c.execute('''UPDATE users SET
                 total_transactions = total_transactions + 1,
                 avg_amount = (avg_amount * total_transactions + ?) / (total_transactions + 1)
                 WHERE user_id=?''', (amount, user_id))
    if is_fraud:
        c.execute("UPDATE users SET fraud_count = fraud_count+1 WHERE user_id=?", (user_id,))
    conn.commit()
    conn.close()

def update_transaction_feedback(txn_id, confirmed):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE transactions SET confirmed_by_user=? WHERE txn_id=?",
              (1 if confirmed else 0, txn_id))
    conn.commit()
    conn.close()

def get_user_transactions(user_id, limit=20):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''SELECT * FROM transactions WHERE user_id=?
                 ORDER BY timestamp DESC LIMIT ?''', (user_id, limit))
    rows = c.fetchall()
    conn.close()
    return rows

def get_all_users():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT user_id, name, email, avg_amount, usual_location, total_transactions, fraud_count FROM users")
    rows = c.fetchall()
    conn.close()
    return rows

def log_model_performance(auc, acc, rec, total):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO model_log VALUES (NULL,?,?,?,?,?)",
              (auc, acc, rec, total, datetime.datetime.now().isoformat()))
    conn.commit()
    conn.close()

# ─────────────────────────────────────────────
# OTP SYSTEM
# ─────────────────────────────────────────────
def generate_otp(user_id):
    otp = ''.join(random.choices(string.digits, k=6))
    otp_store[user_id] = {
        "code": otp,
        "expires": datetime.datetime.now() + datetime.timedelta(minutes=5)
    }
    return otp

def verify_otp(user_id, entered_otp):
    record = otp_store.get(user_id)
    if not record:
        return False, "No OTP found. Please request a new one."
    if datetime.datetime.now() > record["expires"]:
        del otp_store[user_id]
        return False, "OTP expired. Please request a new one."
    if record["code"] == entered_otp.strip():
        del otp_store[user_id]
        return True, "OTP verified successfully!"
    return False, "Incorrect OTP. Please try again."

# ─────────────────────────────────────────────
# EMAIL SYSTEM
# ─────────────────────────────────────────────
def send_email_alert(to_email, user_name, amount, location, otp, prob):
    """Send fraud alert email using credentials from .env file."""
    sender = SENDER_EMAIL or os.environ.get("FRAUD_SENDER_EMAIL", "")
    passw  = SENDER_PASS  or os.environ.get("FRAUD_SENDER_PASS",  "")

    if not sender or not passw:
        return False, "Email credentials not configured. Add FRAUD_SENDER_EMAIL and FRAUD_SENDER_PASS to your .env file."

    subject = "FRAUD ALERT — Suspicious Transaction Detected"
    body = f"""Dear {user_name},

We detected a SUSPICIOUS transaction on your account:

  Amount    : ${amount:.2f}
  Location  : {location}
  Time      : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
  Risk Score: {prob:.1%}

Your OTP Verification Code: {otp}

If this was YOU: Enter this OTP in the app to approve the transaction.
If this was NOT you: Click DENY in the app to block this transaction immediately.

Stay Safe,
Fraud Detection System
"""
    try:
        msg = MIMEMultipart()
        msg['From']    = sender
        msg['To']      = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender, passw)
        server.send_message(msg)
        server.quit()
        return True, "Email sent successfully!"
    except Exception as e:
        return False, f"Email failed: {str(e)}"

# ─────────────────────────────────────────────
# LOCATION & TIME ANOMALY DETECTION
# ─────────────────────────────────────────────
LOCATIONS = ["Karachi", "Lahore", "Islamabad", "Peshawar",
             "London", "Dubai", "New York", "Tokyo", "Uganda", "Moscow"]

HIGH_RISK_LOCATIONS = ["Uganda", "Moscow", "Unknown"]

def detect_location_anomaly(user_location, txn_location):
    if txn_location in HIGH_RISK_LOCATIONS:
        return True, 0.4
    if txn_location != user_location:
        return True, 0.2
    return False, 0.0

def detect_time_anomaly(usual_start, usual_end, txn_hour):
    if txn_hour < usual_start or txn_hour > usual_end:
        return True, 0.15
    return False, 0.0

def detect_amount_anomaly(avg_amount, txn_amount):
    if avg_amount > 0 and txn_amount > avg_amount * 3:
        return True, 0.2
    return False, 0.0

def calculate_risk_score(base_prob, loc_boost, time_boost, amt_boost):
    total = base_prob + loc_boost + time_boost + amt_boost
    return min(total, 1.0)

# ─────────────────────────────────────────────
# INPUT VALIDATION HELPERS
# ─────────────────────────────────────────────
def validate_float(value, field_name, min_val=None, max_val=None):
    """Validate a float input. Returns (value, error_message)."""
    try:
        v = float(value)
        if min_val is not None and v < min_val:
            return None, f"{field_name} must be at least {min_val}."
        if max_val is not None and v > max_val:
            return None, f"{field_name} must be at most {max_val}."
        return v, None
    except ValueError:
        return None, f"{field_name} must be a valid number."

# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
class FraudDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Fraud Detection System v3.0")
        self.root.geometry("1300x850")
        self.root.configure(bg=BG_DARK)
        self.root.resizable(True, True)
        self.current_txn_id  = None
        self.current_user_id = None
        self.build_ui()

        # Try to load saved models on startup
        if load_models():
            self.status_dot.config(text="● Model Loaded from Disk", fg=GREEN)
            self.log("  Saved model loaded successfully from disk!")
            self.log("  You can predict immediately or retrain.\n")

    def build_ui(self):
        header = tk.Frame(self.root, bg=BG_CARD, pady=12)
        header.pack(fill='x')
        tk.Label(header, text="💳  AI-Based Self-Learning Fraud Detection System",
                 font=("Segoe UI", 18, "bold"), bg=BG_CARD, fg=TEXT).pack(side='left', padx=20)
        self.status_dot = tk.Label(header, text="● Not Trained",
                                   font=("Segoe UI", 11), bg=BG_CARD, fg=RED)
        self.status_dot.pack(side='right', padx=10)
        self.time_label = tk.Label(header, text="", font=("Segoe UI", 10),
                                   bg=BG_CARD, fg=TEXT_DIM)
        self.time_label.pack(side='right', padx=20)
        self.update_clock()

        style = ttk.Style()
        style.theme_use('default')
        style.configure("TNotebook", background=BG_DARK, borderwidth=0)
        style.configure("TNotebook.Tab", background=BG_CARD, foreground=TEXT_DIM,
                        padding=[14, 8], font=("Segoe UI", 10))
        style.map("TNotebook.Tab",
                  background=[("selected", BG_INPUT)],
                  foreground=[("selected", TEXT)])

        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill='both', expand=True, padx=10, pady=8)

        self.tab_train   = tk.Frame(self.nb, bg=BG_DARK)
        self.tab_predict = tk.Frame(self.nb, bg=BG_DARK)
        self.tab_db      = tk.Frame(self.nb, bg=BG_DARK)
        self.tab_alerts  = tk.Frame(self.nb, bg=BG_DARK)
        self.tab_charts  = tk.Frame(self.nb, bg=BG_DARK)
        self.tab_report  = tk.Frame(self.nb, bg=BG_DARK)

        self.nb.add(self.tab_train,   text="  🏋️  Train  ")
        self.nb.add(self.tab_predict, text="  🔍  Predict  ")
        self.nb.add(self.tab_db,      text="  🗄️  Database  ")
        self.nb.add(self.tab_alerts,  text="  🚨  Alerts  ")
        self.nb.add(self.tab_charts,  text="  📊  Charts  ")
        self.nb.add(self.tab_report,  text="  📋  Report  ")

        self.build_train_tab()
        self.build_predict_tab()
        self.build_db_tab()
        self.build_alerts_tab()
        self.build_charts_tab()
        self.build_report_tab()

    def update_clock(self):
        now = datetime.datetime.now().strftime("%d %b %Y  %H:%M:%S")
        self.time_label.config(text=now)
        self.root.after(1000, self.update_clock)

    # ─────────────────────────────────────────
    # TAB 1 — TRAIN
    # ─────────────────────────────────────────
    def build_train_tab(self):
        left = tk.Frame(self.tab_train, bg=BG_CARD, width=300)
        left.pack(side='left', fill='y', padx=(10,5), pady=10)
        left.pack_propagate(False)

        tk.Label(left, text="⚙️  Training Settings",
                 font=("Segoe UI", 13, "bold"), bg=BG_CARD, fg=TEXT).pack(pady=(20,15), padx=20, anchor='w')

        self.lbl(left, "Total Transactions")
        self.n_var = tk.IntVar(value=28000)
        self.slider(left, self.n_var, 5000, 50000)

        self.lbl(left, "Fraud Percentage (%)")
        self.fp_var = tk.DoubleVar(value=1.7)
        self.slider(left, self.fp_var, 0.5, 10.0, res=0.1)

        self.lbl(left, "Test Split (%)")
        self.ts_var = tk.IntVar(value=20)
        self.slider(left, self.ts_var, 10, 40)

        self.lbl(left, "Model")
        self.mc_var = tk.StringVar(value="Both")
        for v in ["Both", "Random Forest", "Logistic Regression"]:
            tk.Radiobutton(left, text=v, variable=self.mc_var, value=v,
                           bg=BG_CARD, fg=TEXT, selectcolor=BG_INPUT,
                           activebackground=BG_CARD,
                           font=("Segoe UI", 10)).pack(anchor='w', padx=25, pady=2)

        tk.Button(left, text="🚀  TRAIN MODEL",
                  font=("Segoe UI", 12, "bold"),
                  bg=ACCENT, fg="white", relief='flat',
                  cursor='hand2', pady=10,
                  command=self.start_training).pack(fill='x', padx=20, pady=(15,5))

        tk.Button(left, text="💾  Save Model",
                  font=("Segoe UI", 10),
                  bg=ACCENT2, fg="white", relief='flat',
                  cursor='hand2', pady=7,
                  command=self.save_model_action).pack(fill='x', padx=20, pady=3)

        tk.Button(left, text="📂  Load Model",
                  font=("Segoe UI", 10),
                  bg=BG_INPUT, fg=TEXT, relief='flat',
                  cursor='hand2', pady=7,
                  command=self.load_model_action).pack(fill='x', padx=20, pady=3)

        self.progress = ttk.Progressbar(left, mode='indeterminate')
        self.progress.pack(fill='x', padx=20, pady=8)

        right = tk.Frame(self.tab_train, bg=BG_DARK)
        right.pack(side='right', fill='both', expand=True, padx=(5,10), pady=10)

        tk.Label(right, text="📜  Training Log", font=("Segoe UI", 13, "bold"),
                 bg=BG_DARK, fg=TEXT).pack(anchor='w', padx=10, pady=(10,5))

        self.log_box = scrolledtext.ScrolledText(
            right, font=("Consolas", 10),
            bg=BG_INPUT, fg=GREEN,
            insertbackground=TEXT, relief='flat', borderwidth=5)
        self.log_box.pack(fill='both', expand=True, padx=10, pady=5)

        sf = tk.Frame(right, bg=BG_DARK)
        sf.pack(fill='x', padx=10, pady=8)
        self.cv = {}
        cards = [("ROC-AUC (RF)", "rf_auc", ACCENT2),
                 ("Recall (RF)",  "rf_rec", GREEN),
                 ("Accuracy",     "rf_acc", YELLOW),
                 ("Transactions", "total",  PURPLE)]
        for i, (lbl_txt, key, col) in enumerate(cards):
            card = tk.Frame(sf, bg=BG_CARD, padx=15, pady=10)
            card.grid(row=0, column=i, padx=5, sticky='nsew')
            sf.columnconfigure(i, weight=1)
            tk.Label(card, text=lbl_txt, font=("Segoe UI", 9), bg=BG_CARD, fg=TEXT_DIM).pack()
            v = tk.StringVar(value="—")
            self.cv[key] = v
            tk.Label(card, textvariable=v, font=("Segoe UI", 17, "bold"),
                     bg=BG_CARD, fg=col).pack()

    def save_model_action(self):
        if not trained:
            messagebox.showwarning("No Model", "Please train a model first!")
            return
        if save_models():
            messagebox.showinfo("Saved", f"Models saved to '{MODEL_DIR}/' folder.\nThey will auto-load next time you open the app.")
            self.log("  Models saved to disk successfully.")
        else:
            messagebox.showerror("Error", "Failed to save models.")

    def load_model_action(self):
        if load_models():
            self.status_dot.config(text="● Model Loaded from Disk", fg=GREEN)
            self.log("  Models loaded from disk successfully.")
            messagebox.showinfo("Loaded", "Models loaded successfully! You can now make predictions.")
        else:
            messagebox.showwarning("Not Found", f"No saved models found in '{MODEL_DIR}/' folder.\nPlease train a model first.")

    def lbl(self, p, t):
        tk.Label(p, text=t, font=("Segoe UI", 10), bg=BG_CARD, fg=TEXT_DIM).pack(
            anchor='w', padx=20, pady=(10,2))

    def slider(self, p, var, fr, to, res=1):
        f = tk.Frame(p, bg=BG_CARD)
        f.pack(fill='x', padx=20, pady=2)
        tk.Label(f, textvariable=var, font=("Segoe UI", 10, "bold"),
                 bg=BG_CARD, fg=TEXT, width=6).pack(side='right')
        tk.Scale(f, variable=var, from_=fr, to=to, orient='horizontal',
                 resolution=res, bg=BG_CARD, fg=TEXT,
                 troughcolor=BG_INPUT, highlightthickness=0,
                 showvalue=False).pack(side='left', fill='x', expand=True)

    # ─────────────────────────────────────────
    # TAB 2 — PREDICT
    # ─────────────────────────────────────────
    def build_predict_tab(self):
        outer = tk.Frame(self.tab_predict, bg=BG_DARK)
        outer.pack(fill='both', expand=True, padx=30, pady=15)

        tk.Label(outer, text="🔍  Real-Time Transaction Prediction",
                 font=("Segoe UI", 16, "bold"), bg=BG_DARK, fg=TEXT).pack(pady=(5,3))
        tk.Label(outer, text="Select user, enter transaction details, and check fraud probability",
                 font=("Segoe UI", 10), bg=BG_DARK, fg=TEXT_DIM).pack(pady=(0,15))

        form = tk.Frame(outer, bg=BG_CARD, padx=30, pady=20)
        form.pack(fill='x')

        # Row 1 — User + Amount
        r1 = tk.Frame(form, bg=BG_CARD)
        r1.pack(fill='x', pady=6)
        tk.Label(r1, text="Select User", font=("Segoe UI", 11),
                 bg=BG_CARD, fg=TEXT, width=18, anchor='w').pack(side='left')
        self.user_var = tk.StringVar(value="USR001 — Ali Hassan")
        users = ["USR001 — Ali Hassan", "USR002 — Sara Ahmed",
                 "USR003 — Zeeshan Khan", "USR004 — Fatima Malik"]
        ttk.Combobox(r1, textvariable=self.user_var, values=users,
                     font=("Segoe UI", 11), width=25,
                     state='readonly').pack(side='left', padx=10)
        tk.Label(r1, text="Amount ($)", font=("Segoe UI", 11),
                 bg=BG_CARD, fg=TEXT, width=12, anchor='w').pack(side='left', padx=(20,0))
        self.amt_entry = tk.Entry(r1, font=("Segoe UI", 12),
                                   bg=BG_INPUT, fg=TEXT, insertbackground=TEXT,
                                   relief='flat', width=12)
        self.amt_entry.insert(0, "250.00")
        self.amt_entry.pack(side='left', padx=10, ipady=5)

        # Row 2 — Location + Time
        r2 = tk.Frame(form, bg=BG_CARD)
        r2.pack(fill='x', pady=6)
        tk.Label(r2, text="Transaction Location", font=("Segoe UI", 11),
                 bg=BG_CARD, fg=TEXT, width=18, anchor='w').pack(side='left')
        self.loc_var = tk.StringVar(value="Karachi")
        ttk.Combobox(r2, textvariable=self.loc_var, values=LOCATIONS,
                     font=("Segoe UI", 11), width=18).pack(side='left', padx=10)
        tk.Label(r2, text="Hour (0-23)", font=("Segoe UI", 11),
                 bg=BG_CARD, fg=TEXT, width=12, anchor='w').pack(side='left', padx=(20,0))
        self.hour_var = tk.IntVar(value=datetime.datetime.now().hour)
        tk.Spinbox(r2, from_=0, to=23, textvariable=self.hour_var,
                   font=("Segoe UI", 12), bg=BG_INPUT, fg=TEXT,
                   buttonbackground=BG_CARD, relief='flat', width=6).pack(side='left', padx=10)

        # Row 3 — V features (all 28 in scrollable grid)
        r3 = tk.Frame(form, bg=BG_CARD)
        r3.pack(fill='x', pady=6)
        tk.Label(r3, text="V Features (V1-V28) — 0 = average behaviour. Positive/negative values indicate anomalies.",
                 font=("Segoe UI", 10), bg=BG_CARD, fg=TEXT_DIM).pack(anchor='w', pady=(5,5))

        vr = tk.Frame(form, bg=BG_CARD)
        vr.pack(fill='x', pady=4)
        self.v_ents = []
        for i in range(28):
            col_frame = tk.Frame(vr, bg=BG_CARD)
            col_frame.grid(row=i//7, column=i%7, padx=5, pady=3)
            tk.Label(col_frame, text=f"V{i+1}", font=("Segoe UI", 8),
                     bg=BG_CARD, fg=TEXT_DIM).pack()
            e = tk.Entry(col_frame, font=("Segoe UI", 9),
                         bg=BG_INPUT, fg=TEXT, insertbackground=TEXT,
                         relief='flat', width=7)
            e.insert(0, "0.0")
            e.pack(ipady=4)
            self.v_ents.append(e)

        # Buttons
        br = tk.Frame(form, bg=BG_CARD)
        br.pack(fill='x', pady=15)
        tk.Button(br, text="🔍  CHECK TRANSACTION",
                  font=("Segoe UI", 12, "bold"),
                  bg=ACCENT2, fg="white", relief='flat',
                  cursor='hand2', pady=8, padx=20,
                  command=self.predict_transaction).pack(side='left')
        tk.Button(br, text="🎲  Random Transaction",
                  font=("Segoe UI", 11), bg=BG_INPUT, fg=TEXT,
                  relief='flat', cursor='hand2', pady=8, padx=20,
                  command=self.random_txn).pack(side='left', padx=8)
        tk.Button(br, text="👁️  View User Profile",
                  font=("Segoe UI", 11), bg=PURPLE, fg="white",
                  relief='flat', cursor='hand2', pady=8, padx=20,
                  command=self.show_user_profile).pack(side='left', padx=8)
        tk.Button(br, text="🔄  Clear Fields",
                  font=("Segoe UI", 11), bg=BG_INPUT, fg=TEXT,
                  relief='flat', cursor='hand2', pady=8, padx=20,
                  command=self.clear_predict_fields).pack(side='left', padx=8)

        # Result Area
        res_frame = tk.Frame(outer, bg=BG_DARK)
        res_frame.pack(fill='x', pady=10)

        self.res_label = tk.Label(res_frame, text="",
                                   font=("Segoe UI", 20, "bold"),
                                   bg=BG_DARK, fg=TEXT)
        self.res_label.pack(pady=8)

        self.prob_label = tk.Label(res_frame, text="",
                                    font=("Segoe UI", 12), bg=BG_DARK, fg=TEXT_DIM)
        self.prob_label.pack()

        self.prob_canvas = tk.Canvas(res_frame, height=30,
                                      bg=BG_CARD, highlightthickness=0)
        self.prob_canvas.pack(fill='x', padx=60, pady=8)

        self.anomaly_frame = tk.Frame(res_frame, bg=BG_DARK)
        self.anomaly_frame.pack(fill='x', padx=60, pady=5)

        # OTP Panel
        self.otp_frame = tk.Frame(outer, bg=BG_CARD, padx=20, pady=15)
        self.otp_label_title = tk.Label(self.otp_frame,
                                         text="🔐  OTP Verification Required",
                                         font=("Segoe UI", 13, "bold"),
                                         bg=BG_CARD, fg=YELLOW)
        self.otp_label_title.pack()
        self.otp_info = tk.Label(self.otp_frame, text="",
                                  font=("Segoe UI", 10), bg=BG_CARD, fg=TEXT_DIM)
        self.otp_info.pack(pady=5)

        otp_row = tk.Frame(self.otp_frame, bg=BG_CARD)
        otp_row.pack(pady=8)
        tk.Label(otp_row, text="Enter OTP:", font=("Segoe UI", 11),
                 bg=BG_CARD, fg=TEXT).pack(side='left')
        self.otp_entry = tk.Entry(otp_row, font=("Segoe UI", 14, "bold"),
                                   bg=BG_INPUT, fg=YELLOW, insertbackground=YELLOW,
                                   relief='flat', width=10, justify='center')
        self.otp_entry.pack(side='left', padx=10, ipady=5)

        otp_btns = tk.Frame(self.otp_frame, bg=BG_CARD)
        otp_btns.pack(pady=5)
        tk.Button(otp_btns, text="✅  APPROVE",
                  font=("Segoe UI", 11, "bold"),
                  bg=GREEN, fg="white", relief='flat',
                  cursor='hand2', pady=6, padx=20,
                  command=lambda: self.verify_otp_action(True)).pack(side='left', padx=5)
        tk.Button(otp_btns, text="❌  DENY / BLOCK",
                  font=("Segoe UI", 11, "bold"),
                  bg=RED, fg="white", relief='flat',
                  cursor='hand2', pady=6, padx=20,
                  command=lambda: self.verify_otp_action(False)).pack(side='left', padx=5)

    def clear_predict_fields(self):
        self.amt_entry.delete(0, 'end')
        self.amt_entry.insert(0, "250.00")
        for e in self.v_ents:
            e.delete(0, 'end')
            e.insert(0, "0.0")
        self.loc_var.set("Karachi")
        self.hour_var.set(datetime.datetime.now().hour)
        self.res_label.config(text="")
        self.prob_label.config(text="")
        self.otp_frame.pack_forget()
        for w in self.anomaly_frame.winfo_children():
            w.destroy()

    # ─────────────────────────────────────────
    # TAB 3 — DATABASE
    # ─────────────────────────────────────────
    def build_db_tab(self):
        tk.Label(self.tab_db, text="🗄️  Database — Users & Transactions",
                 font=("Segoe UI", 14, "bold"), bg=BG_DARK, fg=TEXT).pack(pady=10)

        btn_row = tk.Frame(self.tab_db, bg=BG_DARK)
        btn_row.pack(pady=5)
        for txt, col, cmd in [
            ("👥  Load Users",        ACCENT2, self.load_users),
            ("📋  Load Transactions", PURPLE,  self.load_transactions),
            ("➕  Add New User",      ACCENT,  self.add_user_dialog),
            ("🔄  Refresh",           BG_INPUT, self.load_users),
            ("📤  Export CSV",        YELLOW,   self.export_csv),
        ]:
            tk.Button(btn_row, text=txt, font=("Segoe UI", 10),
                      bg=col, fg="white" if col != BG_INPUT else TEXT,
                      relief='flat', cursor='hand2', pady=6, padx=12,
                      command=cmd).pack(side='left', padx=4)

        tbl_frame = tk.Frame(self.tab_db, bg=BG_DARK)
        tbl_frame.pack(fill='both', expand=True, padx=10, pady=5)

        style = ttk.Style()
        style.configure("Custom.Treeview",
                        background=BG_INPUT, foreground=TEXT,
                        fieldbackground=BG_INPUT, rowheight=28,
                        font=("Segoe UI", 10))
        style.configure("Custom.Treeview.Heading",
                        background=BG_CARD, foreground=TEXT,
                        font=("Segoe UI", 10, "bold"))
        style.map("Custom.Treeview", background=[("selected", ACCENT2)])

        self.db_tree = ttk.Treeview(tbl_frame, style="Custom.Treeview")
        scrollbar = ttk.Scrollbar(tbl_frame, orient="vertical",
                                   command=self.db_tree.yview)
        self.db_tree.configure(yscrollcommand=scrollbar.set)
        self.db_tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        self.load_users()

    def export_csv(self):
        """Export the currently displayed table to CSV."""
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            title="Save as CSV"
        )
        if not path:
            return
        try:
            cols = self.db_tree['columns']
            rows = [self.db_tree.item(child)['values'] for child in self.db_tree.get_children()]
            df = pd.DataFrame(rows, columns=cols)
            df.to_csv(path, index=False)
            messagebox.showinfo("Exported", f"Data exported to:\n{path}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    def load_users(self):
        self.db_tree.delete(*self.db_tree.get_children())
        self.db_tree['columns'] = ("ID","Name","Email","Avg Amount","Location","Total Txns","Fraud Count")
        self.db_tree.column("#0", width=0, stretch=False)
        widths = [80, 140, 180, 100, 100, 90, 90]
        for col, w in zip(self.db_tree['columns'], widths):
            self.db_tree.column(col, width=w, anchor='center')
            self.db_tree.heading(col, text=col)
        for row in get_all_users():
            self.db_tree.insert("", "end", values=(
                row[0], row[1], row[2],
                f"${row[3]:.0f}", row[4], row[5], row[6]
            ))

    def load_transactions(self):
        self.db_tree.delete(*self.db_tree.get_children())
        cols = ("TxnID","UserID","Amount","Location","Hour","Fraud%","Fraud","Confirmed","Time")
        self.db_tree['columns'] = cols
        self.db_tree.column("#0", width=0, stretch=False)
        for col in cols:
            self.db_tree.column(col, width=110, anchor='center')
            self.db_tree.heading(col, text=col)
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT * FROM transactions ORDER BY timestamp DESC LIMIT 50")
        for row in c.fetchall():
            conf = {-1: "Pending", 0: "Denied", 1: "Approved"}.get(row[7], "?")
            fraud_lbl = "FRAUD" if row[6] else "Legit"
            self.db_tree.insert("", "end", values=(
                row[0][:8]+"...", row[1], f"${row[2]:.0f}",
                row[3], row[4], f"{row[5]:.1%}",
                fraud_lbl, conf, row[8][:16]
            ))
        conn.close()

    def add_user_dialog(self):
        win = tk.Toplevel(self.root)
        win.title("Add New User")
        win.geometry("400x350")
        win.configure(bg=BG_CARD)
        win.grab_set()

        tk.Label(win, text="➕  Add New User", font=("Segoe UI", 13, "bold"),
                 bg=BG_CARD, fg=TEXT).pack(pady=15)

        fields = [("User ID", "USR005"), ("Name", ""), ("Email", ""),
                  ("Avg Amount ($)", "1000"), ("Usual Location", "Karachi")]
        entries = {}
        for label, default in fields:
            row = tk.Frame(win, bg=BG_CARD)
            row.pack(fill='x', padx=20, pady=5)
            tk.Label(row, text=label, font=("Segoe UI", 10),
                     bg=BG_CARD, fg=TEXT_DIM, width=18, anchor='w').pack(side='left')
            e = tk.Entry(row, font=("Segoe UI", 11), bg=BG_INPUT, fg=TEXT,
                         insertbackground=TEXT, relief='flat')
            e.insert(0, default)
            e.pack(side='left', fill='x', expand=True, ipady=4)
            entries[label] = e

        def save():
            # Validate inputs
            uid = entries["User ID"].get().strip()
            name = entries["Name"].get().strip()
            email = entries["Email"].get().strip()
            if not uid or not name or not email:
                messagebox.showerror("Validation Error", "User ID, Name, and Email are required.")
                return
            amt_val, err = validate_float(entries["Avg Amount ($)"].get(), "Avg Amount", min_val=0)
            if err:
                messagebox.showerror("Validation Error", err)
                return
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            try:
                c.execute('''INSERT INTO users VALUES (?,?,?,?,?,9,21,0,0,?)''', (
                    uid, name, email, amt_val,
                    entries["Usual Location"].get(),
                    datetime.datetime.now().isoformat()
                ))
                conn.commit()
                messagebox.showinfo("Success", "User added successfully!")
                win.destroy()
                self.load_users()
            except sqlite3.IntegrityError:
                messagebox.showerror("Error", f"User ID '{uid}' already exists.")
            except Exception as e:
                messagebox.showerror("Error", str(e))
            finally:
                conn.close()

        tk.Button(win, text="💾  Save User",
                  font=("Segoe UI", 11, "bold"),
                  bg=ACCENT, fg="white", relief='flat',
                  cursor='hand2', pady=8,
                  command=save).pack(pady=15, padx=20, fill='x')

    # ─────────────────────────────────────────
    # TAB 4 — ALERTS
    # ─────────────────────────────────────────
    def build_alerts_tab(self):
        tk.Label(self.tab_alerts, text="🚨  Alert Center — Email & OTP System",
                 font=("Segoe UI", 14, "bold"), bg=BG_DARK, fg=TEXT).pack(pady=10)

        cfg = tk.Frame(self.tab_alerts, bg=BG_CARD, padx=20, pady=15)
        cfg.pack(fill='x', padx=10, pady=5)

        tk.Label(cfg, text="📧  Email Configuration (Gmail)",
                 font=("Segoe UI", 12, "bold"), bg=BG_CARD, fg=TEXT).pack(anchor='w', pady=(0,10))

        # Show current config status
        env_status = "✅ Loaded from .env file" if SENDER_EMAIL else "⚠️ Not configured — enter below or create .env file"
        tk.Label(cfg, text=f"Status: {env_status}",
                 font=("Segoe UI", 9), bg=BG_CARD,
                 fg=GREEN if SENDER_EMAIL else YELLOW).pack(anchor='w', pady=(0,8))

        r1 = tk.Frame(cfg, bg=BG_CARD)
        r1.pack(fill='x', pady=4)
        tk.Label(r1, text="Sender Email:", font=("Segoe UI", 10),
                 bg=BG_CARD, fg=TEXT_DIM, width=18, anchor='w').pack(side='left')
        self.email_from = tk.Entry(r1, font=("Segoe UI", 11),
                                    bg=BG_INPUT, fg=TEXT, insertbackground=TEXT,
                                    relief='flat', width=35)
        self.email_from.insert(0, SENDER_EMAIL or "your_email@gmail.com")
        self.email_from.pack(side='left', padx=10, ipady=4)

        r2 = tk.Frame(cfg, bg=BG_CARD)
        r2.pack(fill='x', pady=4)
        tk.Label(r2, text="App Password:", font=("Segoe UI", 10),
                 bg=BG_CARD, fg=TEXT_DIM, width=18, anchor='w').pack(side='left')
        self.email_pass = tk.Entry(r2, font=("Segoe UI", 11),
                                    bg=BG_INPUT, fg=TEXT, insertbackground=TEXT,
                                    relief='flat', width=35, show="*")
        self.email_pass.insert(0, SENDER_PASS or "")
        self.email_pass.pack(side='left', padx=10, ipady=4)

        tk.Label(cfg,
                 text="Tip: Create a .env file with FRAUD_SENDER_EMAIL=you@gmail.com and FRAUD_SENDER_PASS=yourpassword",
                 font=("Segoe UI", 9), bg=BG_CARD, fg=TEXT_DIM).pack(anchor='w', pady=5)

        tk.Button(cfg, text="💾  Save Credentials to .env",
                  font=("Segoe UI", 10), bg=ACCENT, fg="white",
                  relief='flat', cursor='hand2', pady=6, padx=15,
                  command=self.save_email_credentials).pack(anchor='w', pady=5)

        btn_row = tk.Frame(cfg, bg=BG_CARD)
        btn_row.pack(fill='x', pady=8)
        tk.Button(btn_row, text="📧  Send Test Email",
                  font=("Segoe UI", 10), bg=ACCENT2, fg="white",
                  relief='flat', cursor='hand2', pady=6, padx=15,
                  command=self.test_email).pack(side='left', padx=5)
        tk.Button(btn_row, text="🔐  Generate Test OTP",
                  font=("Segoe UI", 10), bg=YELLOW, fg=BG_DARK,
                  relief='flat', cursor='hand2', pady=6, padx=15,
                  command=self.test_otp).pack(side='left', padx=5)

        tk.Label(self.tab_alerts, text="📋  Alert Log",
                 font=("Segoe UI", 12, "bold"), bg=BG_DARK, fg=TEXT).pack(anchor='w', padx=15, pady=(10,5))

        self.alert_log = scrolledtext.ScrolledText(
            self.tab_alerts, font=("Consolas", 10),
            bg=BG_INPUT, fg=YELLOW,
            insertbackground=TEXT, relief='flat', borderwidth=5)
        self.alert_log.pack(fill='both', expand=True, padx=10, pady=5)
        self.alert_log.insert('end', "Alert log will appear here when fraud is detected...\n")

    def save_email_credentials(self):
        """Save email credentials securely to .env file."""
        email = self.email_from.get().strip()
        passw = self.email_pass.get().strip()
        if not email or not passw:
            messagebox.showwarning("Missing Info", "Please enter both email and password.")
            return
        with open(".env", "w") as f:
            f.write(f"FRAUD_SENDER_EMAIL={email}\n")
            f.write(f"FRAUD_SENDER_PASS={passw}\n")
        global SENDER_EMAIL, SENDER_PASS
        SENDER_EMAIL = email
        SENDER_PASS  = passw
        messagebox.showinfo("Saved", "Credentials saved to .env file.\nThey will auto-load next time.")
        self.log_alert("Email credentials saved to .env file.")

    def test_email(self):
        sender = self.email_from.get().strip()
        passw  = self.email_pass.get().strip()
        if not sender or not passw:
            messagebox.showwarning("Missing Credentials",
                                   "Enter your Gmail and App Password above,\nor save them to .env file.")
            return
        self.log_alert("Sending test email...")
        otp = generate_otp("TEST_USER")
        ok, msg = send_email_alert(sender, "Test User", 999.99, "Test Location", otp, 0.95)
        self.log_alert(f"Test email result: {msg}")
        if ok:
            messagebox.showinfo("Success", "Test email sent successfully!")
        else:
            messagebox.showerror("Failed", msg)

    def test_otp(self):
        otp = generate_otp("TEST_USER")
        self.log_alert(f"Test OTP Generated: {otp} (expires in 5 minutes)")
        messagebox.showinfo("Test OTP", f"Generated OTP: {otp}\n\nIn real system this would be emailed to user.\nOTP expires in 5 minutes.")

    def log_alert(self, msg):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.alert_log.insert('end', f"[{ts}]  {msg}\n")
        self.alert_log.see('end')

    # ─────────────────────────────────────────
    # TAB 5 — CHARTS
    # ─────────────────────────────────────────
    def build_charts_tab(self):
        self.charts_inner = tk.Frame(self.tab_charts, bg=BG_DARK)
        self.charts_inner.pack(fill='both', expand=True)
        tk.Label(self.charts_inner, text="Train the model first to see charts",
                 font=("Segoe UI", 14), bg=BG_DARK, fg=TEXT_DIM).pack(expand=True)

    # ─────────────────────────────────────────
    # TAB 6 — REPORT
    # ─────────────────────────────────────────
    def build_report_tab(self):
        tk.Label(self.tab_report, text="📋  Full Classification Report",
                 font=("Segoe UI", 14, "bold"), bg=BG_DARK, fg=TEXT).pack(pady=10)

        btn_row = tk.Frame(self.tab_report, bg=BG_DARK)
        btn_row.pack(pady=5)
        tk.Button(btn_row, text="💾  Save Chart as PNG",
                  font=("Segoe UI", 10), bg=ACCENT, fg="white",
                  relief='flat', cursor='hand2', pady=6, padx=15,
                  command=self.save_chart).pack(side='left', padx=5)
        tk.Button(btn_row, text="🔄  Retrain Model",
                  font=("Segoe UI", 10), bg=BG_INPUT, fg=TEXT,
                  relief='flat', cursor='hand2', pady=6, padx=15,
                  command=self.start_training).pack(side='left', padx=5)
        tk.Button(btn_row, text="📤  Export Report as TXT",
                  font=("Segoe UI", 10), bg=ACCENT2, fg="white",
                  relief='flat', cursor='hand2', pady=6, padx=15,
                  command=self.export_report).pack(side='left', padx=5)

        self.report_box = scrolledtext.ScrolledText(
            self.tab_report, font=("Consolas", 10),
            bg=BG_INPUT, fg=TEXT,
            insertbackground=TEXT, relief='flat', borderwidth=10)
        self.report_box.pack(fill='both', expand=True, padx=10, pady=10)

    def export_report(self):
        content = self.report_box.get(1.0, 'end')
        if not content.strip():
            messagebox.showwarning("Empty", "No report to export. Train the model first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")],
            title="Export Report"
        )
        if path:
            with open(path, "w") as f:
                f.write(content)
            messagebox.showinfo("Exported", f"Report saved to:\n{path}")

    # ─────────────────────────────────────────
    # TRAINING LOGIC
    # ─────────────────────────────────────────
    def start_training(self):
        self.progress.start(10)
        self.log("=" * 55)
        self.log("  AI Fraud Detection System — Training Started")
        self.log("=" * 55)
        threading.Thread(target=self.train_model, daemon=True).start()

    def log(self, msg):
        self.log_box.insert('end', msg + "\n")
        self.log_box.see('end')
        self.root.update_idletasks()

    def train_model(self):
        global model_rf, model_lr, scaler_amt, scaler_tim
        global X_test_g, y_test_g, df_global, trained

        # Safe defaults
        rauc = rrec = racc = rrep = None

        try:
            np.random.seed(42)
            n_total = self.n_var.get()
            fp      = self.fp_var.get() / 100
            n_fraud = int(n_total * fp)
            n_legit = n_total - n_fraud

            self.log(f"\n  Dataset     : {n_total:,} transactions")
            self.log(f"  Fraud cases : {n_fraud} ({fp*100:.1f}%)")
            self.log(f"  Legit cases : {n_legit:,}")

            self.log("\n  Generating synthetic transaction data...")
            V_l = np.random.randn(n_legit, 28)
            V_f = np.random.randn(n_fraud, 28) + np.random.choice([-2,2], (n_fraud,28))

            df_l = pd.DataFrame(V_l, columns=[f'V{i}' for i in range(1,29)])
            df_l['Time']   = np.random.uniform(0, 172792, n_legit)
            df_l['Amount'] = np.abs(np.random.exponential(88, n_legit))
            df_l['Class']  = 0

            df_f = pd.DataFrame(V_f, columns=[f'V{i}' for i in range(1,29)])
            df_f['Time']   = np.random.uniform(0, 172792, n_fraud)
            df_f['Amount'] = np.abs(np.random.exponential(122, n_fraud))
            df_f['Class']  = 1

            df = pd.concat([df_l, df_f]).sample(frac=1, random_state=42).reset_index(drop=True)
            df_global = df.copy()

            self.log("  Preprocessing — Scaling Amount & Time...")
            sa = StandardScaler()
            st = StandardScaler()
            df['scaled_Amount'] = sa.fit_transform(df[['Amount']])
            df['scaled_Time']   = st.fit_transform(df[['Time']])
            df.drop(['Amount','Time'], axis=1, inplace=True)
            scaler_amt = sa
            scaler_tim = st

            X = df.drop('Class', axis=1)
            y = df['Class']
            ts = self.ts_var.get() / 100
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=ts, random_state=42, stratify=y)
            X_test_g = Xte
            y_test_g = yte

            self.log("  Handling class imbalance — Oversampling fraud cases...")
            fraud_x = Xtr[ytr==1]
            legit_x = Xtr[ytr==0]
            tgt = int(len(legit_x)*0.2)
            oi  = np.random.choice(len(fraud_x), tgt, replace=True)
            Xb  = pd.concat([legit_x, fraud_x.iloc[oi]]).reset_index(drop=True)
            yb  = pd.concat([pd.Series([0]*len(legit_x)), pd.Series([1]*tgt)]).reset_index(drop=True)
            sh  = np.random.permutation(len(Xb))
            Xb  = Xb.iloc[sh].reset_index(drop=True)
            yb  = yb.iloc[sh].reset_index(drop=True)
            self.log(f"  Balanced training size: {len(Xb):,}")

            choice = self.mc_var.get()

            if choice in ["Both", "Logistic Regression"]:
                self.log("  Training Logistic Regression...")
                lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
                lr.fit(Xb, yb)
                model_lr = lr
                lp = lr.predict_proba(Xte)[:,1]
                self.log(f"  LR ROC-AUC : {roc_auc_score(yte, lp):.4f}")

            if choice in ["Both", "Random Forest"]:
                self.log("  Training Random Forest (100 trees)...")
                rf = RandomForestClassifier(n_estimators=100, class_weight='balanced',
                                            random_state=42, n_jobs=-1)
                rf.fit(Xb, yb)
                model_rf = rf
                rp    = rf.predict(Xte)
                rprob = rf.predict_proba(Xte)[:,1]
                rauc  = roc_auc_score(yte, rprob)
                rrec  = recall_score(yte, rp)
                racc  = accuracy_score(yte, rp)
                rrep  = classification_report(yte, rp, target_names=['Legit','Fraud'])
                self.log(f"  RF ROC-AUC : {rauc:.4f}")
                self.log(f"  RF Recall  : {rrec:.4f}")
                self.log(f"  RF Accuracy: {racc:.4f}")
                log_model_performance(rauc, racc, rrec, n_total)

            trained = True
            self.log("\n  Model training complete!")
            self.log("  Auto-saving models to disk...")
            if save_models():
                self.log("  Models saved successfully.")
            self.log("=" * 55)

            self.root.after(0, lambda: self.post_train_update(
                rauc, rrec, racc, rrep, n_total, Xte, yte, choice
            ))

        except Exception as e:
            self.log(f"  ERROR: {e}")
            import traceback
            self.log(traceback.format_exc())
        finally:
            self.root.after(0, self.progress.stop)

    def post_train_update(self, auc, rec, acc, rep, total, Xte, yte, choice):
        self.status_dot.config(text="● Model Trained & Ready", fg=GREEN)
        if auc is not None:
            self.cv['rf_auc'].set(f"{auc:.3f}")
            self.cv['rf_rec'].set(f"{rec:.1%}")
            self.cv['rf_acc'].set(f"{acc:.1%}")
        self.cv['total'].set(f"{total:,}")

        self.report_box.delete(1.0, 'end')
        if rep:
            self.report_box.insert('end', "="*55+"\n  RANDOM FOREST REPORT\n"+"="*55+"\n"+rep+"\n")
        if model_lr:
            lp     = model_lr.predict(Xte)
            lr_rep = classification_report(yte, lp, target_names=['Legit','Fraud'])
            self.report_box.insert('end', "="*55+"\n  LOGISTIC REGRESSION REPORT\n"+"="*55+"\n"+lr_rep)

        self.build_charts_display(Xte, yte, choice)

    # ─────────────────────────────────────────
    # CHARTS DISPLAY
    # ─────────────────────────────────────────
    def build_charts_display(self, Xte, yte, choice):
        for w in self.charts_inner.winfo_children():
            w.destroy()

        fig = plt.Figure(figsize=(14, 8), facecolor=BG_DARK)
        gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.5, wspace=0.4)

        def sax(ax, title):
            ax.set_facecolor(BG_CARD)
            ax.set_title(title, color=TEXT, fontsize=9, fontweight='bold', pad=8)
            ax.tick_params(colors=TEXT, labelsize=8)
            for sp in ax.spines.values():
                sp.set_edgecolor(BORDER)
            ax.xaxis.label.set_color(TEXT)
            ax.yaxis.label.set_color(TEXT)

        ax1 = fig.add_subplot(gs[0,0])
        cnts = df_global['Class'].value_counts()
        ax1.bar(['Legit','Fraud'],[cnts.get(0,0), cnts.get(1,0)], color=[GREEN,RED], edgecolor='white', lw=0.5)
        sax(ax1,"Class Distribution")

        ax2 = fig.add_subplot(gs[0,1])
        la = df_global[df_global['Class']==0]['Amount']
        fa = df_global[df_global['Class']==1]['Amount']
        ax2.hist(la[la<500],bins=40,alpha=0.7,color=GREEN,density=True,label='Legit')
        ax2.hist(fa[fa<500],bins=40,alpha=0.7,color=RED,density=True,label='Fraud')
        ax2.legend(facecolor=BG_CARD,labelcolor=TEXT,fontsize=7)
        sax(ax2,"Amount Distribution")

        if model_rf and choice in ["Both","Random Forest"]:
            rprob = model_rf.predict_proba(Xte)[:,1]
            rpred = model_rf.predict(Xte)

            ax3 = fig.add_subplot(gs[0,2])
            fpr,tpr,_ = roc_curve(yte,rprob)
            auc = roc_auc_score(yte,rprob)
            ax3.plot(fpr,tpr,color=ACCENT2,lw=2,label=f'RF AUC={auc:.3f}')
            ax3.plot([0,1],[0,1],'w--',lw=1,alpha=0.4)
            ax3.fill_between(fpr,tpr,alpha=0.1,color=ACCENT2)
            ax3.legend(facecolor=BG_CARD,labelcolor=TEXT,fontsize=7)
            ax3.grid(True,color=BORDER,alpha=0.4)
            sax(ax3,"ROC Curve — Random Forest")

            ax4 = fig.add_subplot(gs[0,3])
            cm = confusion_matrix(yte,rpred)
            ax4.imshow(cm,cmap='Blues',aspect='auto')
            for i in range(2):
                for j in range(2):
                    ax4.text(j,i,str(cm[i,j]),ha='center',va='center',
                             color='white',fontsize=13,fontweight='bold')
            ax4.set_xticks([0,1]); ax4.set_yticks([0,1])
            ax4.set_xticklabels(['Legit','Fraud'],color=TEXT,fontsize=8)
            ax4.set_yticklabels(['Legit','Fraud'],color=TEXT,fontsize=8)
            sax(ax4,"Confusion Matrix — RF")

            ax5 = fig.add_subplot(gs[1,0:2])
            fi  = pd.Series(model_rf.feature_importances_, index=Xte.columns)
            top = fi.sort_values(ascending=False)[:12]
            clrs = [RED if v>0.03 else ACCENT2 for v in top.values]
            ax5.barh(top.index[::-1],top.values[::-1],color=clrs[::-1])
            sax(ax5,"Feature Importances (Top 12)")
            ax5.grid(True,axis='x',color=BORDER,alpha=0.4)

        if model_lr and choice in ["Both","Logistic Regression"]:
            lprob = model_lr.predict_proba(Xte)[:,1]
            ax6   = fig.add_subplot(gs[1,2])
            fpr,tpr,_ = roc_curve(yte,lprob)
            auc = roc_auc_score(yte,lprob)
            ax6.plot(fpr,tpr,color=YELLOW,lw=2,label=f'LR AUC={auc:.3f}')
            ax6.plot([0,1],[0,1],'w--',lw=1,alpha=0.4)
            ax6.fill_between(fpr,tpr,alpha=0.1,color=YELLOW)
            ax6.legend(facecolor=BG_CARD,labelcolor=TEXT,fontsize=7)
            sax(ax6,"ROC Curve — Logistic Regression")
            ax6.grid(True,color=BORDER,alpha=0.4)

        ax7 = fig.add_subplot(gs[1,3])
        if model_rf and choice in ["Both","Random Forest"]:
            rprob = model_rf.predict_proba(Xte)[:,1]
            p,r,_ = precision_recall_curve(yte,rprob)
            ap    = average_precision_score(yte,rprob)
            ax7.plot(r,p,color=ACCENT2,lw=2,label=f'RF AP={ap:.3f}')
        if model_lr and choice in ["Both","Logistic Regression"]:
            lprob = model_lr.predict_proba(Xte)[:,1]
            p,r,_ = precision_recall_curve(yte,lprob)
            ap    = average_precision_score(yte,lprob)
            ax7.plot(r,p,color=YELLOW,lw=2,label=f'LR AP={ap:.3f}')
        ax7.legend(facecolor=BG_CARD,labelcolor=TEXT,fontsize=7)
        sax(ax7,"Precision-Recall Curve")
        ax7.grid(True,color=BORDER,alpha=0.4)

        fig.suptitle("AI Fraud Detection — Model Performance Dashboard",
                     color=TEXT,fontsize=13,fontweight='bold')
        self.current_fig = fig

        canvas = FigureCanvasTkAgg(fig, master=self.charts_inner)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    # ─────────────────────────────────────────
    # PREDICTION LOGIC
    # ─────────────────────────────────────────
    def predict_transaction(self):
        if not trained:
            messagebox.showwarning("Not Trained", "Please train the model first!")
            return

        # Validate amount
        amt_val, err = validate_float(self.amt_entry.get(), "Amount", min_val=0.01)
        if err:
            messagebox.showerror("Input Error", err)
            return

        # Validate all V features
        v_vals = []
        for i, e in enumerate(self.v_ents):
            v, err = validate_float(e.get(), f"V{i+1}")
            if err:
                messagebox.showerror("Input Error", err)
                return
            v_vals.append(v)

        try:
            uid  = self.user_var.get().split(" — ")[0]
            user = get_user(uid)
            if not user:
                messagebox.showerror("Error", "User not found!")
                return

            location = self.loc_var.get()
            hour     = self.hour_var.get()

            loc_anom,  loc_boost  = detect_location_anomaly(user[4], location)
            time_anom, time_boost = detect_time_anomaly(user[5], user[6], hour)
            amt_anom,  amt_boost  = detect_amount_anomaly(user[3], amt_val)

            sa = scaler_amt.transform([[amt_val]])[0][0]
            st = scaler_tim.transform([[hour * 7200]])[0][0]

            # Use all 28 V features
            feats = v_vals + [sa, st]
            cols  = [f'V{i}' for i in range(1,29)] + ['scaled_Amount','scaled_Time']
            Xinp  = pd.DataFrame([feats], columns=cols)

            model     = model_rf if model_rf else model_lr
            base_prob = model.predict_proba(Xinp)[0][1]
            final_prob = calculate_risk_score(base_prob, loc_boost, time_boost, amt_boost)
            is_fraud   = final_prob > 0.5

            txn_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=12))
            save_transaction(txn_id, uid, amt_val, location, hour, final_prob, int(is_fraud))
            self.current_txn_id  = txn_id
            self.current_user_id = uid

            self.show_result(final_prob, is_fraud, amt_val, location, hour,
                             loc_anom, time_anom, amt_anom,
                             loc_boost, time_boost, amt_boost,
                             user, txn_id)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def show_result(self, prob, is_fraud, amount, location, hour,
                    loc_anom, time_anom, amt_anom,
                    lb, tb, ab, user, txn_id):

        for w in self.anomaly_frame.winfo_children():
            w.destroy()

        if is_fraud:
            self.res_label.config(text="🚨  FRAUDULENT TRANSACTION DETECTED!", fg=RED)
        elif prob > 0.3:
            self.res_label.config(text="⚠️  SUSPICIOUS — Verify Required", fg=YELLOW)
        else:
            self.res_label.config(text="✅  LEGITIMATE TRANSACTION", fg=GREEN)

        self.prob_label.config(
            text=f"Risk Score: {prob:.1%}  |  Amount: ${amount:.2f}  |  Location: {location}  |  Hour: {hour}:00",
            fg=TEXT_DIM)

        self.draw_bar(prob)

        tags = tk.Frame(self.anomaly_frame, bg=BG_DARK)
        tags.pack(fill='x', pady=5)

        anomalies = [
            (loc_anom,  f"📍 Location Anomaly ({location} ≠ {user[4]})", lb),
            (time_anom, f"🕐 Time Anomaly (Hour {hour} outside {user[5]}-{user[6]})", tb),
            (amt_anom,  f"💰 Amount Anomaly (${amount:.0f} vs avg ${user[3]:.0f})", ab),
        ]
        for detected, msg, boost in anomalies:
            col = RED if detected else GREEN
            lbl_txt = "DETECTED" if detected else "Normal"
            tk.Label(tags,
                     text=f"  {'🔴' if detected else '🟢'}  {msg}  [{lbl_txt}] +{boost:.0%}  ",
                     font=("Segoe UI", 9), bg=BG_CARD,
                     fg=col, padx=8, pady=4).pack(side='left', padx=4)

        if is_fraud or prob > 0.4:
            otp = generate_otp(self.current_user_id)
            self.otp_frame.pack(fill='x', padx=60, pady=10)
            self.otp_info.config(
                text=f"OTP: {otp}  (Expires in 5 min — In real system this would be emailed to {user[2]})")
            self.log_alert(f"FRAUD ALERT — User: {user[1]} | Amount: ${amount:.2f} | Location: {location} | Risk: {prob:.1%}")
            self.log_alert(f"OTP Generated: {otp} → Would be sent to {user[2]}")
            self.log_alert(f"Transaction ID: {txn_id}")

            # Attempt real email if credentials are set
            if SENDER_EMAIL and SENDER_PASS:
                threading.Thread(
                    target=lambda: self._send_alert_email(user, amount, location, otp, prob),
                    daemon=True
                ).start()
        else:
            self.otp_frame.pack_forget()
            self.log_alert(f"APPROVED — User: {user[1]} | Amount: ${amount:.2f} | Risk: {prob:.1%}")

    def _send_alert_email(self, user, amount, location, otp, prob):
        ok, msg = send_email_alert(user[2], user[1], amount, location, otp, prob)
        self.root.after(0, lambda: self.log_alert(f"Email: {msg}"))

    def draw_bar(self, prob):
        self.prob_canvas.update_idletasks()
        w = self.prob_canvas.winfo_width()
        h = 30
        self.prob_canvas.delete('all')
        self.prob_canvas.create_rectangle(0,0,w,h,fill=BG_INPUT,outline='')
        fw    = int(w * prob)
        color = RED if prob > 0.6 else YELLOW if prob > 0.3 else GREEN
        self.prob_canvas.create_rectangle(0,0,fw,h,fill=color,outline='')
        self.prob_canvas.create_text(w//2,h//2,
                                      text=f"{prob:.1%} Risk Score",
                                      fill='white',font=("Segoe UI",10,"bold"))

    def verify_otp_action(self, approved):
        if not self.current_user_id or not self.current_txn_id:
            return
        if approved:
            entered = self.otp_entry.get().strip()
            ok, msg = verify_otp(self.current_user_id, entered)
            if ok:
                update_transaction_feedback(self.current_txn_id, True)
                self.log_alert("OTP VERIFIED — Transaction APPROVED by user")
                messagebox.showinfo("Approved", "OTP Correct!\nTransaction Approved.")
                self.otp_frame.pack_forget()
                self.res_label.config(text="✅  Transaction Approved by User", fg=GREEN)
            else:
                self.log_alert(f"OTP Failed: {msg}")
                messagebox.showerror("Wrong OTP", msg)
        else:
            update_transaction_feedback(self.current_txn_id, False)
            self.log_alert("Transaction DENIED & BLOCKED by user — Marked as confirmed fraud")
            messagebox.showwarning("Blocked", "Transaction Blocked!\nMarked as confirmed fraud.")
            self.otp_frame.pack_forget()
            self.res_label.config(text="🚫  Transaction BLOCKED — Fraud Confirmed", fg=RED)

    def random_txn(self):
        is_fraud = random.random() < 0.3
        self.amt_entry.delete(0,'end')
        self.amt_entry.insert(0, str(round(
            random.uniform(500,5000) if is_fraud else random.expovariate(1/200), 2)))
        self.loc_var.set(random.choice(HIGH_RISK_LOCATIONS if is_fraud else ["Karachi","Lahore","Islamabad"]))
        self.hour_var.set(random.randint(0,4) if is_fraud else random.randint(10,20))
        v_vals = np.random.randn(28)*2 + (np.random.choice([-3,3],28) if is_fraud else np.zeros(28))
        for i, e in enumerate(self.v_ents):
            e.delete(0,'end')
            e.insert(0, f"{v_vals[i]:.3f}")

    def show_user_profile(self):
        uid  = self.user_var.get().split(" — ")[0]
        user = get_user(uid)
        if not user:
            return
        txns = get_user_transactions(uid, 5)
        info = f"""
USER PROFILE
═══════════════════════════════
ID            : {user[0]}
Name          : {user[1]}
Email         : {user[2]}
Avg Amount    : ${user[3]:.2f}
Usual Location: {user[4]}
Active Hours  : {user[5]}:00 — {user[6]}:00
Total Txns    : {user[7]}
Fraud Count   : {user[8]}
Member Since  : {user[9][:10]}

RECENT TRANSACTIONS (Last 5)
═══════════════════════════════"""
        for t in txns:
            info += f"\n  ${t[2]:.0f}  {t[3]}  {t[4]}:00  Risk:{t[5]:.0%}  {'FRAUD' if t[6] else 'Legit'}"
        messagebox.showinfo(f"Profile — {user[1]}", info)

    # ─────────────────────────────────────────
    # SAVE CHART
    # ─────────────────────────────────────────
    def save_chart(self):
        if not hasattr(self, 'current_fig'):
            messagebox.showwarning("No Chart", "Train the model first!")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png")],
            title="Save Chart"
        )
        if path:
            self.current_fig.savefig(path, dpi=150, bbox_inches='tight',
                                      facecolor=self.current_fig.get_facecolor())
            messagebox.showinfo("Saved!", f"Chart saved as:\n{path}")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Initializing database...")
    init_database()
    print("Launching GUI...")
    root = tk.Tk()
    app  = FraudDetectorApp(root)
    root.mainloop()