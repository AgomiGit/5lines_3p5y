import argparse
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import yfinance as yf
import fear_greed


def parse_args():
    parser = argparse.ArgumentParser(description="產生美股或台股的樂活五線譜儀表板")
    parser.add_argument(
        "ticker",
        nargs="?",
        default="QQQ",
        help="Yahoo Finance 代碼；例如 AAPL、2330（上市）或 6488.TWO（上櫃）",
    )
    return parser.parse_args()


def normalize_ticker(raw_ticker):
    ticker = raw_ticker.strip().upper()
    if not ticker:
        raise ValueError("股票代碼不可為空")
    if ticker.isdigit():
        ticker = f"{ticker}.TW"
    return ticker

# ==========================================
# 1. 數據抓取與樂活五線譜計算
# ==========================================
args = parse_args()
ticker = normalize_ticker(args.ticker)
is_taiwan_stock = ticker.endswith((".TW", ".TWO"))
currency_symbol = "NT$" if is_taiwan_stock else "$"
sentiment_description = (
    "統計學價格軌道與 CNN 美股市場情緒參考"
    if is_taiwan_stock
    else "統計學價格軌道與美股即時情緒指標"
)
days_window = 875

print(f"正在抓取 {ticker} 的歷史數據...")
df_yf = yf.download(ticker, start="2022-01-01")

if df_yf.empty:
    raise SystemExit(f"找不到 {ticker} 的歷史數據，請確認 Yahoo Finance 股票代碼是否正確")

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

last_date = df.index[-1].strftime('%Y-%m-%d')

# ==========================================
# 2. 將計算好的數據打包成前端 JS 用的 JSON 格式
# ==========================================
chart_data = []
for date_idx, row in df.iterrows():
    chart_data.append({
        "date": date_idx.strftime('%Y-%m-%d'),
        "close": round(row['close'], 2),
        "tl": round(row['TL'], 2),
        "p1sd": round(row['TL_plus_1SD'], 2),
        "p2sd": round(row['TL_plus_2SD'], 2),
        "m1sd": round(row['TL_minus_1SD'], 2),
        "m2sd": round(row['TL_minus_2SD'], 2)
    })
json_data_str = json.dumps(chart_data)

# 提取最新一天的數據作為預設顯示
latest_data = chart_data[-1]

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

if current_fg_score <= 25:
    gauge_color = "#cc0000" # 極度恐慌
elif current_fg_score <= 45:
    gauge_color = "#ff9900" # 恐慌
elif current_fg_score <= 55:
    gauge_color = "#888888" # 中立
elif current_fg_score <= 75:
    gauge_color = "#66cc00" # 貪婪
else:
    gauge_color = "#008800" # 極度貪婪

# ==========================================
# 4. 輸出手機版儀表板網頁
# ==========================================
with open("index.html", "w", encoding="utf-8") as f:
    f.write(f"""<!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{ticker} 投資決策儀表板</title>
        <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 0; padding: 15px; background-color: #f8f9fa; color: #333; display: flex; flex-direction: column; align-items: center; }}
            h1 {{ color: #111; font-size: 24px; margin: 10px 0 5px 0; text-align: center; }}
            h2 {{ color: #6c757d; font-size: 13px; font-weight: normal; margin-bottom: 20px; text-align: center; padding: 0 10px; }}
            
            /* 數據卡片排版 */
            .metric-box {{ display: flex; justify-content: center; gap: 15px; margin-bottom: 20px; flex-wrap: wrap; width: 100%; max-width: 1200px; }}
            .metric {{ background: #fff; padding: 12px 20px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.04); flex: 1; min-width: 140px; text-align: center; }}
            .metric-title {{ font-size: 11px; color: #6c757d; text-transform: uppercase; }}
            .metric-value {{ font-size: 20px; font-weight: bold; color: #111; margin-top: 5px; }}
            
            /* CNN 進度條 */
            .gauge-wrapper {{ background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); width: 100%; max-width: 1200px; margin-bottom: 20px; box-sizing: border-box; }}
            .gauge-title {{ font-size: 14px; font-weight: bold; margin-bottom: 12px; color: #111; }}
            .gauge-bar-bg {{ background: #e9ecef; height: 20px; border-radius: 10px; position: relative; overflow: hidden; display: flex; }}
            .gauge-fill {{ background: {gauge_color}; width: {current_fg_score}%; height: 100%; border-radius: 10px 0 0 10px; }}
            .gauge-text {{ position: absolute; right: 15px; top: 1px; color: #fff; font-weight: bold; font-size: 12px; text-shadow: 1px 1px 2px rgba(0,0,0,0.5); }}
            .gauge-labels {{ display: flex; justify-content: space-between; margin-top: 6px; font-size: 11px; color: #6c757d; font-weight: bold; }}
            
            /* 全寬直向排列佈局 */
            .main-content {{ display: flex; flex-direction: column; width: 100%; max-width: 1200px; gap: 20px; margin-bottom: 20px; }}
            .chart-container {{ background: white; padding: 15px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); min-width: 0; }}
            
            /* 緊湊下置式數據卡 */
            .data-card {{ background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border: 2px solid #e0e6ed; box-sizing: border-box; width: 100%; transition: all 0.3s ease; text-align: left; }}
            .data-card.active {{ border-color: #4A90E2; box-shadow: 0 4px 16px rgba(74, 144, 226, 0.15); }}
            .sidebar-title {{ margin-top: 0; color: #333; border-bottom: 2px solid #eef2f5; padding-bottom: 8px; font-size: 18px; }}
            
            /* 價格大字盒 */
            .price-box {{ background-color: #f1f5f9; padding: 12px 20px; border-radius: 6px; font-size: 20px; font-weight: bold; margin: 15px 0; display: flex; justify-content: space-between; align-items: center; }}
            
            /* 手機自適應網格 */
            .grid-list {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; padding: 0; margin: 0; list-style: none; }}
            .grid-list li {{ padding: 12px; background: #f8f9fa; border-radius: 6px; border-left: 4px solid #ccc; display: flex; flex-direction: column; gap: 4px; }}
            .grid-list li .line-name {{ font-size: 12px; color: #666; font-weight: 500; }}
            .grid-list li .line-val {{ font-size: 16px; font-weight: bold; color: #111; }}
            
            .hint {{ color: #888; font-size: 13px; line-height: 1.6; margin: 0; text-align: center; }}
            
            /* 隱藏 Plotly 預設提示框 */
            .hoverlayer {{ display: none !important; visibility: hidden !important; opacity: 0 !important; }}
            
            .info {{ margin-top: 20px; font-size: 11px; color: #adb5bd; text-align: center; padding: 0 10px; }}
        </style>
    </head>
    <body>
        <h1>{ticker} 🎯 投資決策儀表板</h1>
        <h2>數據驅動看板：{sentiment_description}</h2>
        
        <div class="metric-box">
            <div class="metric">
                <div class="metric-title">{ticker} 最新收盤價</div>
                <div class="metric-value">{currency_symbol}{df['close'].iloc[-1]:.2f}</div>
            </div>
            <div class="metric">
                <div class="metric-title">五線譜中軸 (TL)</div>
                <div class="metric-value" style="color: #008800;">{currency_symbol}{df['TL'].iloc[-1]:.2f}</div>
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
        
        <div class="main-content">
            <div class="chart-container">
                <div style="font-size: 15px; font-weight: bold; text-align: left; margin-bottom: 10px; color: #111;">📈 {ticker} 樂活五線譜趨勢圖</div>
                <div id="plotly-chart"></div>
            </div>
            
            <div id="sidebar-content" class="data-card active">
                <h3 class="sidebar-title" style="color: #4A90E2; border-bottom-color: #4A90E2; margin-bottom: 12px;">📈 歷史數據細節 (📅 {latest_data['date']})</h3>
                <div class="price-box">
                    <span>💰 {ticker} 當日股價:</span>
                    <span style="color: #222;">{currency_symbol}{latest_data['close']:.2f}</span>
                </div>
                <ul class="grid-list">
                    <li style="border-left-color: #dc3545;"><span class="line-name">🔴 極樂觀線 (+2SD)</span><span class="line-val">{currency_symbol}{latest_data['p2sd']:.2f}</span></li>
                    <li style="border-left-color: #ffc107;"><span class="line-name">🟠 相對樂觀 (+1SD)</span><span class="line-val">{currency_symbol}{latest_data['p1sd']:.2f}</span></li>
                    <li style="border-left-color: #28a745;"><span class="line-name">🔵 趨勢中軸 (TL)</span><span class="line-val">{currency_symbol}{latest_data['tl']:.2f}</span></li>
                    <li style="border-left-color: #007bff;"><span class="line-name">🟢 相對悲觀 (-1SD)</span><span class="line-val">{currency_symbol}{latest_data['m1sd']:.2f}</span></li>
                    <li style="border-left-color: #6f42c1;"><span class="line-name">🟣 極悲觀線 (-2SD)</span><span class="line-val">{currency_symbol}{latest_data['m2sd']:.2f}</span></li>
                </ul>
            </div>
        </div>

        <div class="info">
            <p>最後更新日期：{last_date} | 本網頁由 GitHub Actions 雲端虛擬機於美股收盤後天天自動執行更新</p>
        </div>

        <script>
            const stockData = {json_data_str};
            const dates = stockData.map(d => d.date);
            const closes = stockData.map(d => d.close);
            
            const data = [
                {{ x: dates, y: closes, name: '{ticker} 真實收盤價', type: 'scatter', mode: 'lines', line: {{color: '#000000', width: 2}} }},
                {{ x: dates, y: stockData.map(d => d.p2sd), name: '極樂觀線 (+2SD)', type: 'scatter', line: {{color: '#dc3545', width: 1.5, dash: 'dash'}} }},
                {{ x: dates, y: stockData.map(d => d.p1sd), name: '相對樂觀線 (+1SD)', type: 'scatter', line: {{color: '#ffc107', width: 1.5, dash: 'dash'}} }},
                {{ x: dates, y: stockData.map(d => d.tl), name: '趨勢中軸線 (TL)', type: 'scatter', line: {{color: '#28a745', width: 2}} }},
                {{ x: dates, y: stockData.map(d => d.m1sd), name: '相對悲觀線 (-1SD)', type: 'scatter', line: {{color: '#007bff', width: 1.5, dash: 'dash'}} }},
                {{ x: dates, y: stockData.map(d => d.m2sd), name: '極悲觀線 (-2SD)', type: 'scatter', line: {{color: '#6f42c1', width: 1.5, dash: 'dash'}} }}
            ];

            const layout = {{
                hovermode: 'x unified',
                margin: {{ r: 5, l: 35, t: 10, b: 30 }},
                height: 400,
                legend: {{ orientation: 'h', y: -0.2, x: 0.5, xanchor: 'center' }},
                xaxis: {{ type: 'date', gridcolor: '#f0f0f0', fixedrange: true }},
                yaxis: {{ gridcolor: '#f0f0f0', fixedrange: true }},
                plot_bgcolor: '#ffffff',
                paper_bgcolor: '#ffffff'
            }};

            const config = {{
                scrollZoom: false,
                displayModeBar: false,
                responsive: true
            }};

            const chartDiv = document.getElementById('plotly-chart');
            Plotly.newPlot(chartDiv, data, layout, config);

            // 點選圖表資料點時切換日期
            chartDiv.on('plotly_click', function(dataEvent){{
                const pointIndex = dataEvent.points[0].pointIndex;
                const selectedData = stockData[pointIndex];
                if(selectedData) {{
                    const sidebar = document.getElementById('sidebar-content');
                    sidebar.className = "data-card active";
                    
                    sidebar.innerHTML = `
                        <h3 class="sidebar-title" style="color: #4A90E2; border-bottom-color: #4A90E2; margin-bottom: 12px;">📈 歷史數據細節 (📅 ` + selectedData.date + `)</h3>
                        <div class="price-box">
                            <span>💰 {ticker} 當日股價:</span>
                            <span style="color: #222;">{currency_symbol}` + selectedData.close.toFixed(2) + `</span>
                        </div>
                        <ul class="grid-list">
                            <li style="border-left-color: #dc3545;"><span class="line-name">🔴 極樂觀線 (+2SD)</span><span class="line-val">{currency_symbol}` + selectedData.p2sd.toFixed(2) + `</span></li>
                            <li style="border-left-color: #ffc107;"><span class="line-name">🟠 相對樂觀 (+1SD)</span><span class="line-val">{currency_symbol}` + selectedData.p1sd.toFixed(2) + `</span></li>
                            <li style="border-left-color: #28a745;"><span class="line-name">🔵 趨勢中軸 (TL)</span><span class="line-val">{currency_symbol}` + selectedData.tl.toFixed(2) + `</span></li>
                            <li style="border-left-color: #007bff;"><span class="line-name">🟢 相對悲觀 (-1SD)</span><span class="line-val">{currency_symbol}` + selectedData.m1sd.toFixed(2) + `</span></li>
                            <li style="border-left-color: #6f42c1;"><span class="line-name">🟣 極悲觀線 (-2SD)</span><span class="line-val">{currency_symbol}` + selectedData.m2sd.toFixed(2) + `</span></li>
                        </ul>
                    `;
                }}
            }});
        </script>
    </body>
    </html>
    """)
print("視覺化儀表板網頁 index.html 產生成功！")
