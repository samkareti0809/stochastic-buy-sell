import sqlite3
import requests
import io
import time
from datetime import date
import os
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

DB_NAME = "master_nse_warehouse.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stock_ohlcv (
            symbol TEXT,
            date TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            PRIMARY KEY (symbol, date)
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# --- 1. Indicator Functions ---
def kalman_filter(src, sharpness=25.0, k_val=1.0):
    kfilt = np.zeros(len(src), dtype=float)
    velocity = np.zeros(len(src), dtype=float)
    kfilt[0] = src.iloc[0]
    velocity[0] = 0.0
    factor = np.sqrt(sharpness * k_val / 100.0)
    k_factor = k_val / 100.0
    src_vals = src.values
    for i in range(1, len(src)):
        distance = src_vals[i] - kfilt[i-1]
        error = kfilt[i-1] + distance * factor
        velocity[i] = velocity[i-1] + distance * k_factor
        kfilt[i] = error + velocity[i]
    return pd.Series(kfilt, index=src.index)

def calculate_ma(series, ma_type, length):
    if ma_type == 'SMA':
        return series.rolling(length).mean()
    elif ma_type == 'EMA':
        return series.ewm(span=length, adjust=False).mean()
    elif ma_type == 'TEMA':
        e = series.ewm(span=length, adjust=False).mean()
        e2 = e.ewm(span=length, adjust=False).mean()
        e3 = e2.ewm(span=length, adjust=False).mean()
        return 3 * (e - e2) + e3
    else:
        return series.ewm(span=length, adjust=False).mean()

def calculate_stochastic(high, low, close, period_k=14, smooth_k=1, period_d=3):
    lowest_low = low.rolling(period_k).min()
    highest_high = high.rolling(period_k).max()
    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    k = stoch_k.rolling(smooth_k).mean()
    return k.rolling(period_d).mean()

def compute_srocst(df):
    kfilt = kalman_filter(df['Close'])
    roc = 100 * (kfilt - kfilt.shift(9)) / kfilt.shift(9)
    d = calculate_stochastic(df['High'], df['Low'], df['Close'])
    blend = (roc + d) / 2
    return calculate_ma(blend, 'TEMA', 12)

# --- 2. Fetching & Expanding the Universe ---
print("Fetching comprehensive NSE symbol universe...")
url = "https://raw.githubusercontent.com/kprohith/nse-stock-analysis/master/ind_nifty500list.csv"
try:
    res = requests.get(url)
    df_symbols = pd.read_csv(io.StringIO(res.text))
    raw_symbols = df_symbols['Symbol'].dropna().unique().tolist()
    nse_universe = [str(s).strip() + ".NS" for s in raw_symbols]
except Exception as e:
    nse_universe = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'ICICIBANK.NS', 'HDFCBANK.NS', 'SBIN.NS']

print(f"Total symbols queued: {len(nse_universe)}\n")

start_date = "2023-01-01"
end_date = date.today().strftime("%Y-%m-%d")
holding_days = 4
stop_loss_pct = 0.01

# --- 3. Batch-Wise ETL Pipeline with Incremental SQLite Commits ---
BATCH_SIZE = 150
total_batches = (len(nse_universe) + BATCH_SIZE - 1) // BATCH_SIZE

print(f"--- STARTING BATCH-WISE INGESTION ({total_batches} Batches Total) ---")

for batch_idx in range(total_batches):
    start_idx = batch_idx * BATCH_SIZE
    end_idx = min(start_idx + BATCH_SIZE, len(nse_universe))
    batch_tickers = nse_universe[start_idx:end_idx]

    print(f"\nProcessing Batch {batch_idx + 1} of {total_batches} (Tickers {start_idx+1} to {end_idx})...")

    conn = sqlite3.connect(DB_NAME)
    batch_success = 0

    for ticker in batch_tickers:
        try:
            df_d = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False, progress=False)
            if isinstance(df_d.columns, pd.MultiIndex):
                df_d.columns = df_d.columns.get_level_values(0)

            if len(df_d) < 200:
                continue

            avg_turnover = (df_d['Close'] * df_d['Volume']).tail(30).mean()
            latest_close = df_d['Close'].iloc[-1]

            if latest_close < 30 or avg_turnover < 5000000:
                continue

            df_d = df_d.reset_index()
            date_col = 'Date' if 'Date' in df_d.columns else df_d.columns[0]

            for _, row in df_d.iterrows():
                d_str = str(row[date_col])[:10]
                conn.execute('''
                    INSERT OR REPLACE INTO stock_ohlcv (symbol, date, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (ticker, d_str, float(row['Open']), float(row['High']), float(row['Low']), float(row['Close']), float(row['Volume'])))
            batch_success += 1
        except Exception as e:
            continue

    conn.commit()
    conn.close()
    print(f"Batch {batch_idx + 1} completed. Successfully stored {batch_success} liquid stocks.")
    time.sleep(2)

print("\nAll batches successfully ingested and cached in SQLite warehouse!\n")

# --- 4. Stage 3: Strategy Backtest Across Stored Warehouse ---
print("--- STAGE 3: Running Strategy Backtest Across Warehouse ---")
results = []
conn = sqlite3.connect(DB_NAME)
cursor = conn.cursor()
cursor.execute("SELECT DISTINCT symbol FROM stock_ohlcv")
qualified_symbols = [row[0] for row in cursor.fetchall()]

for ticker in qualified_symbols:
    try:
        query = f"SELECT date, open, high, low, close, volume FROM stock_ohlcv WHERE symbol = '{ticker}' ORDER BY date ASC"
        df = pd.read_sql(query, conn, parse_dates=['date'])
        df.set_index('date', inplace=True)

        df['Blend'] = compute_srocst(df)
        df['Blend_Prev'] = df['Blend'].shift(1)
        df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['SMA_200'] = df['Close'].rolling(200).mean()

        srocst_buy = (df['Blend'] > df['Blend_Prev']) & (df['Blend_Prev'] <= df['Blend'].shift(2))
        ema_touch = (df['Low'] <= df['EMA_10'] * 1.05) | (df['Low'] <= df['EMA_20'] * 1.05)
        uptrend = df['Close'] > df['SMA_200']

        signal_day = uptrend & srocst_buy & ema_touch
        green_confirmation = (df['Close'] > df['Open']) & (df['Close'] > df['Close'].shift(1))
        df['Valid_Entry'] = signal_day.shift(1) & green_confirmation

        positions = np.zeros(len(df))
        market_returns = df['Close'].pct_change().fillna(0)
        trade_returns = []

        i = 0
        while i < len(df):
            if df['Valid_Entry'].iloc[i]:
                entry_idx = i
                signal_low = df['Low'].iloc[entry_idx - 1]
                stop_loss_price = signal_low * (1 - stop_loss_pct)

                exit_idx = min(entry_idx + holding_days, len(df) - 1)
                actual_exit_idx = exit_idx

                for day in range(entry_idx, exit_idx + 1):
                    positions[day] = 1
                    if df['Low'].iloc[day] <= stop_loss_price:
                        actual_exit_idx = day
                        positions[day:exit_idx+1] = 0
                        break

                t_ret = (df['Close'].iloc[actual_exit_idx] / df['Open'].iloc[entry_idx]) - 1
                trade_returns.append(t_ret)
                i = exit_idx + 1
            else:
                positions[i] = 0
                i += 1

        df['Position'] = positions
        df['Strategy_Return'] = market_returns * df['Position']

        bh_ret = (1 + market_returns).prod() - 1
        strat_ret = (1 + df['Strategy_Return']).prod() - 1

        total_trades = len(trade_returns)
        winning_trades = sum(1 for r in trade_returns if r > 0)
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0

        results.append({
            'Stock': ticker.replace('.NS', ''),
            'Buy & Hold (%)': round(bh_ret * 100, 2),
            'Strategy Return (%)': round(strat_ret * 100, 2),
            'Total Trades': total_trades,
            'Win Rate (%)': round(win_rate, 2)
        })
    except Exception as e:
        continue

conn.close()

# --- 5. Display & Telegram Notification Payload ---
message_lines = ["MARKET-WIDE ELITE MASTER WATCHLIST (WIN RATE >= 80%)", "--------------------------------------------------"]

if results:
    res_df = pd.DataFrame(results)
    elite_market_wide = res_df[res_df['Win Rate (%)'] >= 80].sort_values(by='Win Rate (%)', ascending=False)

    if not elite_market_wide.empty:
        table_str = elite_market_wide.to_string(index=False)
        print(table_str)
        message_lines.append(table_str)
    else:
        msg = "No stocks met the >= 80% win rate threshold."
        print(msg)
        message_lines.append(msg)
else:
    msg = "Backtest completed with no valid transactions recorded."
    print(msg)
    message_lines.append(msg)

final_message = "\n".join(message_lines)

# --- 6. Send Telegram Notification ---
def send_telegram_message(text):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("Telegram credentials not found in environment variables. Skipping notification.")
        return
    
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": f"```\n{text}\n```",
        "parse_mode": "Markdown"
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("Telegram notification sent successfully!")
        else:
            print(f"Failed to send Telegram notification: {response.text}")
    except Exception as e:
        print(f"Error sending Telegram message: {e}")

send_telegram_message(final_message)
