import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.linear_model import LinearRegression
import yfinance as yf
import fear_greed  # 引入最新穩定的 CNN API 套件
import mpld3

# 1. 解決繪圖中文字型碎碼問題
plt.rcParams['axes.unicode_minus'] = False
try:
    mpl.rc('font', family='Microsoft JhengHei')
except:
    pass

# ==========================================
# 2. 數據抓取與五線譜計算
# ==========================================
ticker = "QQQ"
days_window = 875

print(f"正在抓取 {ticker} 的歷史數據...")
df_yf = yf.download(ticker, start="2022-01-01")

if isinstance(df_yf.columns, pd.MultiIndex):
    df_yf.columns = df_yf.columns.droplevel(1)

df_yf.index = pd.to_datetime(df_yf.index)
df = df_yf[['Close']].rename(columns={'Close': 'close'}).tail(days_window).copy()

df['X'] = np.arange(len(df))
X = df['X'].values.reshape(-1, 1)
Y = df['close'].values

model = LinearRegression()
model.fit(X, Y)
df['TL'] = model.predict(X)

residuals = Y - df['TL'].values
sd = np.std(residuals)

df['TL_plus_2SD'] = df['TL'] + (2 * sd)
df['TL_plus_1SD'] = df['TL'] + (1 * sd)
df['TL_minus_1SD'] = df['TL'] - (1 * sd)
df['TL_minus_2SD'] = df['TL'] - (2 * sd)

# 繪製圖表 1：正宗樂活五線譜
fig1, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(df.index, df['close'], label=f'{ticker} 真實收盤價', color='black', linewidth=2)
ax1.plot(df.index, df['TL_plus_2SD'], label='極樂觀線 (+2SD)', color='red', linestyle='--')
ax1.plot(df.index, df['TL_plus_1SD'], label='相對樂觀線 (+1SD)', color='orange', linestyle='--')
ax1.plot(df.index, df['TL'], label='趨勢中軸線 (TL)', color='green', linewidth=2)
ax1.plot(df.index, df['TL_minus_1SD'], label='相對悲觀線 (-1SD)', color='blue', linestyle='--')
ax1.plot(df.index, df['TL_minus_2SD'], label='極悲觀線 (-2SD)', color='purple', linestyle='--')

last_date = df.index[-1].strftime('%Y-%m-%d')
ax1.set_title(f'{ticker} 正宗樂活五線譜 (數據截至: {last_date})', fontsize=14, fontweight='bold')
ax1.set_xlabel('日期 (Date)')
ax1.set_ylabel('價格 (USD)')
ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
ax1.grid(True, alpha=0.3)
plt.tight_layout()

# ==========================================
# 3. 獲取 CNN Fear & Greed Index
# ==========================================
print("正在獲取 CNN Fear & Greed Index 數據...")
fg_data = fear_greed.get()
current_fg_score = fg_data['score']
current_fg_rating = fg_data['rating'].upper()

# 獲取 CNN 內建的一年歷史情緒數據
fg_history = fg_data.get('historical_data', [])
if fg_history:
    df_fg = pd.DataFrame(fg_history)
    df_fg['x'] = pd.to_datetime(df_fg['x'], unit='ms') # 轉換 CNN 時間戳記
    df_fg = df_fg.sort_values('x')
else:
    # 備用方案：如果結構微調，建立一組空資料避免程式崩潰
    df_fg = pd.DataFrame(columns=['x', 'y'])

# 繪製圖表 2：CNN 情緒走勢圖
fig2, ax2 = plt.subplots(figsize=(12, 4))
if not df_fg.empty:
    ax2.plot(df_fg['x'], df_fg['y'], color='purple', linewidth=2, label='Fear & Greed Index')
# 繪製 25, 50, 75 基準線
ax2.axhline(y=25, color='red', linestyle=':', alpha=0.6, label='極度恐慌 (25)')
ax2.axhline(y=50, color='gray', linestyle=':', alpha=0.4)
ax2.axhline(y=75, color='green', linestyle=':', alpha=0.6, label='極度貪婪 (75)')

ax2.set_title(f'CNN Fear & Greed Index 歷史走勢 (最新分數: {current_fg_score} - {current_fg_rating})', fontsize=14, fontweight='bold')
ax2.set_ylim(0, 100)
ax2.set_ylabel('Score (0-100)')
ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))
ax2.grid(True, alpha=0.3)
plt.tight_layout()

# ==========================================
# 4. 將雙圖表整合並輸出為單一 HTML 網頁
# ==========================================
chart1_html = mpld3.fig_to_html(fig1)
chart2_html = mpld3.fig_to_html(fig2)

with open("index.html", "w", encoding="utf-8") as f:
    f.write(f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>{ticker} 投資決策儀表板</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 30px; background-color: #f5f5f5; text-align: center; }}
            .container {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); display: inline-block; margin-bottom: 20px; width: 85%; max-width: 1200px; }}
            h1 {{ color: #333; margin-bottom: 5px; }}
            h2 {{ color: #666; font-size: 16px; margin-bottom: 25px; }}
            .metric-box {{ display: flex; justify-content: center; gap: 40px; margin-bottom: 25px; }}
            .metric {{ background: #fff; padding: 15px 25px; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
            .metric-value {{ font-size: 24px; font-weight: bold; color: #111; margin-top: 5px; }}
            .info {{ margin-top: 20px; font-size: 13px; color: #888; }}
        </style>
    </head>
    <body>
        <h1>{ticker} 🎯 投資決策自動儀表板</h1>
        <h2>結合統計學價格動態與美股情緒指標</h2>
        
        <div class="metric-box">
            <div class="metric">
                <div>{ticker} 最新收盤價</div>
                <div class="metric-value">${df['close'].iloc[-1]:.2f}</div>
            </div>
            <div class="metric">
                <div>五線譜中軸 (TL)</div>
                <div class="metric-value">${df['TL'].iloc[-1]:.2f}</div>
            </div>
            <div class="metric">
                <div>CNN 恐懼與貪婪指數</div>
                <div class="metric-value" style="color: {'#cc0000' if current_fg_score <= 25 else '#008800' if current_fg_score >= 75 else '#333'}">
                    {current_fg_score} ({current_fg_rating})
                </div>
            </div>
        </div>

        <div class="container">
            {chart1_html}
        </div>
        
        <br>
        
        <div class="container">
            {chart2_html}
        </div>

        <div class="info">
            <p>數據更新時間：{last_date} | 網頁由 GitHub Actions 免費雲端伺服器每日自動化編譯生成</p>
        </div>
    </body>
    </html>
    """)
print("包含 CNN 恐懼貪婪指數的雙圖表網頁更新成功！")
