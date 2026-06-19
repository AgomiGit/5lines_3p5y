import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import yfinance as yf
import fear_greed
import json

# ==========================================
# 1. 數據抓取與正宗樂活五線譜計算 (完全保留你滿意的正確版本)
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
# 4. 輸出全新排版網頁 (修復字串解析，並關閉 Hover 遮擋)
# ==========================================
with open("index.html", "w", encoding="utf-8") as f:
    f.write(f"""<!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>{ticker} 投資決策儀表板</title>
        <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
        <style>
            body {{ font-family: 'Helvetica Neue', Arial, sans-serif; margin: 30px; background-color: #f8f9fa; color: #333; display: flex; flex-direction: column; align-items: center; }}
            h1 {{ color: #111; font-size: 28px; margin-bottom: 5px; text-align: center; }}
            h2 {{ color: #6c757d; font-size: 15px; font-weight: normal; margin-bottom: 30px; text-align: center; }}
            
            /* 數據卡片排版 */
            .metric-box {{ display: flex; justify-content: center; gap: 30px; margin-bottom: 30px; flex-wrap: wrap; width: 85%; max-width: 1200px; }}
            .metric {{ background: #fff; padding: 15px 30px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.04); min-width: 180px; text-align: center; }}
            .metric-title {{ font-size: 13px; color: #6c757d; text-transform: uppercase; letter-spacing: 0.5px; }}
            .metric-value {{ font-size: 26px; font-weight: bold; color: #111; margin-top: 8px; }}
            
            /* CNN 進度條 */
            .gauge-wrapper {{ background: white; padding: 25px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); width: 85%; max-width: 1200px; margin-bottom: 25px; box-sizing: border-box; }}
            .gauge-title {{ font-size: 16px; font-weight: bold; margin-bottom: 15px; color: #111; }}
            .gauge-bar-bg {{ background: #e9ecef; height: 24px; border-radius: 12px; position: relative; overflow: hidden; display: flex; }}
            .gauge-fill {{ background: {gauge_color}; width: {current_fg_score}%; height: 100%; border-radius: 12px 0 0 12px; }}
            .gauge-text {{ position: absolute; right: 15px; top: 2px; color: #fff; font-weight: bold; font-size: 14px; text-shadow: 1px 1px 2px rgba(0,0,0,0.5); }}
            .gauge-labels {{ display: flex; justify-content: space-between; margin-top: 8px; font-size: 12px; color: #6c757d; font-weight: bold; }}
            
            /* 主要內容區：左邊圖表，右邊側邊欄 */
            .main-content {{ display: flex; width: 85%; max-width: 1200px; gap: 25px; margin-bottom: 25px; }}
            .chart-container {{ flex: 1; background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); min-width: 0; }}
            .sidebar {{ width: 320px; }}
            .data-card {{ background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border: 2px solid #e0e6ed; box-sizing: border-box; height: 100%; min-height: 450px; transition: all 0.3s ease; text-align: left; }}
            .data-card.active {{ border-color: #4A90E2; box-shadow: 0 4px 16px rgba(74, 144, 226, 0.15); }}
            .sidebar-title {{ margin-top: 0; color: #333; border-bottom: 2px solid #eef2f5; padding-bottom: 8px; font-size: 18px; }}
            .price-box {{ background-color: #f1f5f9; padding: 12px; border-radius: 6px; font-size: 18px; font-weight: bold; margin: 15px 0; display: flex; justify-content: space-between; }}
            ul {{ list-style: none; padding: 0; margin: 0; }}
            li {{ padding: 10px 0; border-bottom: 1px dashed #f0f0f0; display: flex; justify-content: space-between; font-size: 14px; }}
            .hint {{ color: #888; font-size: 14px; line-height: 1.6; }}
            
            .info {{ margin-top: 30px; font-size: 13px; color: #adb5bd; text-align: center; }}
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
        
        <div class="main-content">
            <div class="chart-container">
                <div style="font-size: 16px; font-weight: bold; text-align: left; margin-bottom: 15px; color: #111;">📈 QQQ 正宗樂活五線譜趨勢圖 (點擊圖上任意點查看細節)</div>
                <div id="plotly-chart"></div>
            </div>
            <div class="sidebar">
                <div id="sidebar-content" class="data-card">
                    <h3 class="sidebar-title">📊 點選數據節點</h3>
                    <p class="hint">請用滑鼠點擊左側圖表上 {ticker} 真實收盤價的任意時間點，此處將即時顯示當天的五線譜詳細價位資訊。</p>
                </div>
            </div>
        </div>

        <div class="info">
            <p>最後更新日期：{last_date} | 本網頁由 GitHub Actions 雲端虛擬機於美股收盤後天天自動執行更新</p>
        </div>

        <script>
            const stockData = {json_data_str};
            const dates = stockData.map(d => d.date);
            const closes = stockData.map(d => d.close);
            
            // 修正點 1：移除了 hoverinfo，設定為 'none' 關閉遮擋方塊，保持五線譜絕對乾淨
            const data = [
                {{ x: dates, y: closes, name: '{ticker} 真實收盤價', type: 'scatter', mode: 'lines', hoverinfo: 'none', line: {{color: '#000000', width: 2}} }},
                {{ x: dates, y: stockData.map(d => d.p2sd), name: '極樂觀線 (+2SD)', type: 'scatter', hoverinfo: 'none', line: {{color: '#dc3545', width: 1.5, dash: 'dash'}} }},
                {{ x: dates, y: stockData.map(d => d.p1sd), name: '相對樂觀線 (+1SD)', type: 'scatter', hoverinfo: 'none', line: {{color: '#ffc107', width: 1.5, dash: 'dash'}} }},
                {{ x: dates, y: stockData.map(d => d.tl), name: '趨勢中軸線 (TL)', type: 'scatter', hoverinfo: 'none', line: {{color: '#28a745', width: 2}} }},
                {{ x: dates, y: stockData.map(d => d.m1sd), name: '相對悲觀線 (-1SD)', type: 'scatter', hoverinfo: 'none', line: {{color: '#007bff', width: 1.5, dash: 'dash'}} }},
                {{ x: dates, y: stockData.map(d => d.m2sd), name: '極悲觀線 (-2SD)', type: 'scatter', hoverinfo: 'none', line: {{color: '#6f42c1', width: 1.5, dash: 'dash'}} }}
            ];

            // 修正點 2：將 hovermode 設為 false，避免游標經過時跑出任何提示框
            const layout = {{
                hovermode: false,
                margin: {{ r: 10, l: 40, t: 10, b: 40 }},
                height: 500,
                legend: {{ orientation: 'h', y: -0.15, x: 0.5, xanchor: 'center' }},
                xaxis: {{ type: 'date', gridcolor: '#f0f0f0' }},
                yaxis: {{ gridcolor: '#f0f0f0' }},
                plot_bgcolor: '#ffffff',
                paper_bgcolor: '#ffffff'
            }};

            const chartDiv = document.getElementById('plotly-chart');
            Plotly.newPlot(chartDiv, data, layout);

            // 監聽前端圖表的點擊事件
            chartDiv.on('plotly_click', function(dataEvent){{
                const pointIndex = dataEvent.points[0].pointIndex;
                const selectedData = stockData[pointIndex];
                if(selectedData) {{
                    const sidebar = document.getElementById('sidebar-content');
                    sidebar.className = "data-card active";
                    
                    // 修正點 3：移除轉義的反斜線，改用正確的 JavaScript Template Literals 字串替換
                    sidebar.innerHTML = `
                        <h3 class="sidebar-title" style="color: #4A90E2; border-bottom-color: #4A90E2;">📈 數據細節</h3>
                        <p><b>📅 日期:</b> ` + selectedData.date + `</p>
                        <div class="price-box">
                            <span>💰 {ticker} 股價:</span>
                            <span style="color: #222;">$` + selectedData.close.toFixed(2) + `</span>
                        </div>
                        <h4 style="margin: 10px 0 5px 0; color: #555; font-size: 14px;">五線譜五個價位：</h4>
                        <ul>
                            <li><span style="color:#dc3545; font-weight:bold;">🔴 極樂觀線 (+2SD):</span> <span>$` + selectedData.p2sd.toFixed(2) + `</span></li>
                            <li><span style="color:#ffc107; font-weight:bold;">🟠 相對樂觀 (+1SD):</span> <span>$` + selectedData.p1sd.toFixed(2) + `</span></li>
                            <li><span style="color:#28a745; font-weight:bold;">🔵 趨勢中軸 (TL):</span> <span>$` + selectedData.tl.toFixed(2) + `</span></li>
                            <li><span style="color:#007bff; font-weight:bold;">🟢 相對悲觀 (-1SD):</span> <span>$` + selectedData.m1sd.toFixed(2) + `</span></li>
                            <li><span style="color:#6f42c1; font-weight:bold;">🟣 極悲觀線 (-2SD):</span> <span>$` + selectedData.m2sd.toFixed(2) + `</span></li>
                        </ul>
                    `;
                }}
            }});
        </script>
    </body>
    </html>
    """)
print("視覺化儀表板網頁 index.html 產生成功！")
