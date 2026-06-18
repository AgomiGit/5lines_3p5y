import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.linear_model import LinearRegression
import yfinance as yf
import fear_greed
import mpld3

# 1. 解決繪圖中文字型碎碼問題
plt.rcParams['axes.unicode_minus'] = False
try:
    mpl.rc('font', family='Microsoft JhengHei')
except:
    pass

# ==========================================
# 2. 數據抓取與正宗樂活五線譜計算
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

# 繪製圖表：正宗樂活五線譜 (這是你滿意的正確版本，完全保留)
fig1, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(df.index, df['close'], label=f'{ticker} 真實收盤價', color='black', linewidth=2)
ax1.plot(df.index, df['TL_plus_2SD'], label='極樂觀線 (+2SD)', color='red', linestyle='--')
ax1.plot(df.index, df['TL_plus_1SD'], label='相對樂觀線 (+1SD)', color='orange', linestyle='--')
ax1.plot(df.index, df['TL'], label='趨勢中軸線 (TL)', color='green', linewidth=2)
ax1.plot(df.index, df['TL_minus_1SD'], label='相對悲觀線 (-1SD)', color='blue', linestyle='--')
ax1.plot(df.index, df['TL_minus_2SD'], label='極悲觀線 (-2SD)', color='purple', linestyle='--')

last_date = df.index[-1].strftime('%Y-%m-%d')
ax1.set_title(f'{ticker} 正宗樂活五線譜 (觀測週期: 3.5年直線版，最後更新: {last_date})', fontsize=14, fontweight='bold')
ax1.set_xlabel('日期 (Date)')
ax1.set_ylabel('價格 (USD)')
ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
ax1.grid(True, alpha=0.3)
plt.tight_layout()

chart1_html = mpld3.fig_to_html(fig1)

# ==========================================
# 3. 獲取 CNN Fear & Greed 當前最新即時分數
# ==========================================
print("正在獲取 CNN Fear & Greed 當前分數...")
try:
    fg_data = fear_greed.get()
    current_fg_score = float(fg_data['score'])
    current_fg_rating = fg_data['rating'].upper()
except:
    current_fg_score = 50.0
    current_fg_rating = "ERROR"

# 根據分數決定儀表板指針顏色
if current_fg_score <= 25:
    gauge_color = "#cc0000" # 極度恐慌 - 紅色
elif current_fg_score <= 45:
    gauge_color = "#ff9900" # 恐慌 - 橘色
elif current_fg_score <= 55:
    gauge_color = "#888888" # 中立 - 灰色
elif current_fg_score <= 75:
    gauge_color = "#66cc00" # 貪婪 - 淺綠
else:
    gauge_color = "#008800" # 極度貪婪 - 深綠

# ==========================================
# 4. 輸出全新排版網頁 (包含超精美 HTML5 視覺化情緒計量條)
# ==========================================
with open("index.html", "w", encoding="utf-8") as f:
    f.write(f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>{ticker} 投資決策儀表板</title>
        <style>
            body {{ font-family: 'Helvetica Neue', Arial, sans-serif; margin: 30px; background-color: #f8f9fa; text-align: center; color: #333; }}
            .container {{ background: white; padding: 25px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); display: inline-block; margin-bottom: 25px; width: 85%; max-width: 1100px; }}
            h1 {{ color: #111; font-size: 28px; margin-bottom: 5px; }}
            h2 {{ color: #6c757d; font-size: 15px; font-weight: normal; margin-bottom: 30px; }}
            
            /* 數據卡片排版 */
            .metric-box {{ display: flex; justify-content: center; gap: 30px; margin-bottom: 30px; flex-wrap: wrap; }}
            .metric {{ background: #fff; padding: 15px 30px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.04); min-width: 180px; }}
            .metric-title {{ font-size: 13px; color: #6c757d; text-transform: uppercase; letter-spacing: 0.5px; }}
            .metric-value {{ font-size: 26px; font-weight: bold; color: #111; margin-top: 8px; }}
            
            /* CNN 網頁原生感情緒進度條 */
            .gauge-wrapper {{ background: white; padding: 25px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); display: inline-block; width: 85%; max-width: 1100px; margin-bottom: 25px; text-align: left; box-sizing: border-box; }}
            .gauge-title {{ font-size: 16px; font-weight: bold; margin-bottom: 15px; color: #111; }}
            .gauge-bar-bg {{ background: #e9ecef; height: 24px; border-radius: 12px; position: relative; overflow: hidden; display: flex; }}
            .gauge-fill {{ background: {gauge_color}; width: {current_fg_score}%; height: 100%; border-radius: 12px 0 0 12px; transition: width 1s ease-in-out; }}
            .gauge-text {{ position: absolute; right: 15px; top: 2px; color: #fff; font-weight: bold; font-size: 14px; text-shadow: 1px 1px 2px rgba(0,0,0,0.5); }}
            .gauge-labels {{ display: flex; justify-content: space-between; margin-top: 8px; font-size: 12px; color: #6c757d; font-weight: bold; }}
            
            .info {{ margin-top: 30px; font-size: 13px; color: #adb5bd; }}
        </style>
    </head>
    <body>
        <h1>{ticker} 🎯 投資決策儀表板</h1>
        <h2>數據驅動看板：完美結合統計學價格軌道與美股即時情緒指標</h2>
        
        <div class="metric-box">
            <div class="metric">
                <div class="metric-title">{ticker} 最新收盤價</div>
                <div class="metric-value">${df['close'].iloc[-1]:.2f}</div>
            </div>
            <div class="metric">
                <div class="metric-title">五線譜中軸 (TL)</div>
                <div class="metric-value" style="color: #008800;">${df['TL'].iloc[-1]:.2f}</div>
            </div>
            <div class="metric">
                <div class="metric-title">CNN 即時情緒狀態</div>
                <div class="metric-value" style="color: {gauge_color};">{current_fg_rating}</div>
            </div>
        </div>

        <div class="gauge-wrapper">
            <div class="gauge-title">📊 CNN Fear & Greed Index 即時量表 (當前分數: {current_fg_score})</div>
            <div class="gauge-bar-bg">
                <div class="gauge-fill"></div>
                <span class="gauge-text" style="left: calc({current_fg_score}% - 25px); color: { '#333' if current_fg_score < 10 else '#fff' }; text-shadow: { 'none' if current_fg_score < 10 else '1px 1px 2px rgba(0,0,0,0.5)' };">{current_fg_score}</span>
            </div>
            <div class="gauge-labels">
                <span style="color: #cc0000;">極度恐慌 (0)</span>
                <span style="color: #ff9900;">恐慌 (25)</span>
                <span style="color: #888888;">中立 (50)</span>
                <span style="color: #66cc00;">貪婪 (75)</span>
                <span style="color: #008800;">極度貪婪 (100)</span>
            </div>
        </div>
        
        <br>

        <div class="container">
            <div style="font-size: 16px; font-weight: bold; text-align: left; margin-bottom: 15px; color: #111;">📈 QQQ 正宗樂活五線譜趨勢圖</div>
            {chart1_html}
        </div>

        <div class="info">
            <p>最後更新日期：{last_date} | 本網頁由 GitHub Actions 雲端虛擬機於美股收盤後天天自動執行更新</p>
        </div>
    </body>
    </html>
    """)
print("視覺化儀表板網頁 index.html 產生成功！")
