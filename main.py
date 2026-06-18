import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from openbb import obb
from sklearn.linear_model import LinearRegression
import os

# 1. 解決繪圖中文字型碎碼問題
plt.rcParams['axes.unicode_minus'] = False
try:
    mpl.rc('font', family='Microsoft JhengHei')
except:
    pass

# 2. 數據抓取
ticker = "QQQ"
days_window = 875

print(f"正在抓取 {ticker} 的歷史數據...")
df_obb = obb.equity.price.historical(symbol=ticker, provider="yfinance", start_date="2022-01-01").to_df()
df_obb.index = pd.to_datetime(df_obb.index)
df = df_obb[['close']].tail(days_window).copy()

# 3. 正宗樂活五線譜計算
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

# 4. 繪製圖表
fig, ax = plt.subplots(figsize=(15, 9))
ax.plot(df.index, df['close'], label=f'{ticker} 真實收盤價', color='black', linewidth=2)
ax.plot(df.index, df['TL_plus_2SD'], label='極樂觀線 (+2SD)', color='red', linestyle='--')
ax.plot(df.index, df['TL_plus_1SD'], label='相對樂觀線 (+1SD)', color='orange', linestyle='--')
ax.plot(df.index, df['TL'], label='趨勢中軸線 (TL)', color='green', linewidth=2)
ax.plot(df.index, df['TL_minus_1SD'], label='相對悲觀線 (-1SD)', color='blue', linestyle='--')
ax.plot(df.index, df['TL_minus_2SD'], label='極悲觀線 (-2SD)', color='purple', linestyle='--')

last_date = df.index[-1].strftime('%Y-%m-%d')
ax.set_title(f'{ticker} 正宗樂活五線譜 (自動更新版 - 數據截至: {last_date})', fontsize=16, fontweight='bold')
ax.set_xlabel('日期 (Date)', fontsize=12)
ax.set_ylabel('價格 (USD)', fontsize=12)
ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()

# 5. 【核心修改】不顯示視窗，直接存成 HTML 網頁
# 這裡利用 matplotlib 的內建功能，直接把圖表包裝成一個帶有互動網頁功能的 HTML 檔
import mpld3
html_content = mpld3.fig_to_html(fig)

with open("index.html", "w", encoding="utf-8") as f:
    f.write(f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>{ticker} 樂活五線譜看板</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 30px; background-color: #f5f5f5; text-align: center; }}
            .container {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); display: inline-block; }}
            h1 {{ color: #333; }}
            .info {{ margin-top: 15px; font-size: 14px; color: #666; }}
        </style>
    </head>
    <body>
        <h1>{ticker} 樂活五線譜自動看板</h1>
        <div class="container">
            {html_content}
        </div>
        <div class="info">
            <p>最新收盤價: ${df['close'].iloc[-1]:.2f} | 趨勢中軸 (TL): ${df['TL'].iloc[-1]:.2f}</p>
            <p>本網頁由 GitHub Actions 於每日美股收盤後自動更新</p>
        </div>
    </body>
    </html>
    """)
print("網頁 index.html 產生成功！")
