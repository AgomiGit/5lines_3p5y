import json
import os
import pandas as pd
from openbb import obb

print("正在透過 OpenBB 抓取 QQQ 最新數據...")
# 1. 抓取歷史數據
df = obb.equity.price.historical(symbol="QQQ", provider="yfinance").to_df()
df.index = pd.to_datetime(df.index)

# 2. 計算 3.5 年五線譜 (840 交易日)
window = 840
df["MA"] = df["close"].rolling(window=window).mean()
df["STD"] = df["close"].rolling(window=window).std()

df["悲觀線"] = df["MA"] - 2 * df["STD"]
df["相對悲觀線"] = df["MA"] - 1 * df["STD"]
df["趨勢中線"] = df["MA"]
df["相對樂觀線"] = df["MA"] + 1 * df["STD"]
df["樂觀線"] = df["MA"] + 2 * df["STD"]

# 剔除尚未有均線的舊資料
df_clean = df.dropna().reset_index()

# 3. 轉成前端 JavaScript 陣列所需的 JSON 格式
chart_data = []
for _, row in df_clean.iterrows():
    chart_data.append(
        {
            "date": row["date"].strftime("%Y-%m-%d"),
            "close": round(row["close"], 2),
            "line1": round(row["悲觀線"], 2),
            "line2": round(row["相對悲觀線"], 2),
            "line3": round(row["趨勢中線"], 2),
            "line4": round(row["相對樂觀線"], 2),
            "line5": round(row["樂觀線"], 2),
        }
    )

json_data_str = json.dumps(chart_data)

# 4. 核心：將 Plotly.js 點擊事件與 RWD 側邊欄直接寫入 HTML 模板
# 註：這裡的雙大括號 {{}} 是為了防止 Python f-string 解析出錯，生成為網頁後會變回正常單大括號。
html_template = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QQQ 五線譜動態儀表板</title>
    <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f7fb;
            display: flex;
            justify-content: center;
        }}
        .container {{
            display: flex;
            max-width: 1400px;
            width: 100%;
            background: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            gap: 25px;
        }}
        #chart {{
            flex: 1;
            min-width: 0;
        }}
        .sidebar {{
            width: 320px;
            border-left: 1px solid #eef2f5;
            padding-left: 25px;
            display: flex;
            flex-direction: column;
        }}
        .data-card {{
            border: 2px solid #e0e6ed;
            padding: 20px;
            border-radius: 8px;
            background-color: #fcfdfe;
            box-sizing: border-box;
            transition: all 0.3s ease;
        }}
        .data-card.active {{
            border-color: #4A90E2;
            background-color: #ffffff;
            box-shadow: 0 4px 12px rgba(74, 144, 226, 0.1);
        }}
        h3 {{ margin-top: 0; color: #333; border-bottom: 2px solid #eef2f5; padding-bottom: 8px; }}
        .price-box {{
            background-color: #f1f5f9;
            padding: 12px;
            border-radius: 6px;
            font-size: 18px;
            font-weight: bold;
            margin: 15px 0;
            display: flex;
            justify-content: space-between;
        }}
        ul {{ list-style: none; padding: 0; margin: 0; }}
        li {{
            padding: 10px 0;
            border-bottom: 1px dashed #f0f0f0;
            display: flex;
            justify-content: space-between;
            font-size: 14px;
        }}
        .hint {{ color: #888; font-size: 14px; line-height: 1.6; }}
    </style>
</head>
<body>

<div class="container">
    <div id="chart"></div>

    <div class="sidebar">
        <div id="sidebar-content" class="data-card">
            <h3>📊 點選數據節點</h3>
            <p class="hint">請用滑鼠點擊圖表上 QQQ 收盤價（或五線譜）的任意時間點，此處將即時顯示當天的五線譜詳細價位資訊。</p>
        </div>
    </div>
</div>

<script>
    // GitHub Action 執行時，Python 會把最新計算完的數據直接塞在這裡
    const stockData = {json_data_str};

    const dates = stockData.map(d => d.date);
    const closes = stockData.map(d => d.close);
    const colors = ["#2ca02c", "#94d2bd", "#ff7f0e", "#e29578", "#d62728"];

    // 設定線條數據
    const data = [
        {{ x: dates, y: closes, name: 'QQQ 收盤價', type: 'scatter', mode: 'lines', line: {{color: '#111', width: 2}} }},
        {{ x: dates, y: stockData.map(d => d.line1), name: '悲觀線', type: 'scatter', line: {{color: colors[0], width: 1}} }},
        {{ x: dates, y: stockData.map(d => d.line2), name: '相對悲觀線', type: 'scatter', line: {{color: colors[1], width: 1, dash: 'dash'}} }},
        {{ x: dates, y: stockData.map(d => d.line3), name: '趨勢中線', type: 'scatter', line: {{color: colors[2], width: 1.5}} }},
        {{ x: dates, y: stockData.map(d => d.line4), name: '相對樂觀線', type: 'scatter', line: {{color: colors[3], width: 1, dash: 'dash'}} }},
        {{ x: dates, y: stockData.map(d => d.line5), name: '樂觀線', type: 'scatter', line: {{color: colors[4], width: 1}} }}
    ];

    const layout = {{
        title: 'QQQ 五線譜互動儀表板 (3.5Y 均線)',
        xaxis: {{ title: '日期', type: 'date', rangeslider: {{visible: false}} }},
        yaxis: {{ title: '價格 (USD)', side: 'left' }},
        hovermode: 'x unified',
        margin: {{ r: 20, l: 50, t: 60, b: 50 }},
        legend: {{ orientation: 'h', y: -0.15, x: 0.5, xanchor: 'center' }}
    }};

    const chartDiv = document.getElementById('chart');
    Plotly.newPlot(chartDiv, data, layout);

    // 【核心邏輯】捕捉網頁上的點擊事件，並動態改寫右側 HTML 內容
    chartDiv.on('plotly_click', function(dataEvent){{
        // 抓到點擊點的索引值
        const pointIndex = dataEvent.points[0].pointIndex;
        const selectedData = stockData[pointIndex];

        if(selectedData) {{
            const sidebar = document.getElementById('sidebar-content');
            sidebar.className = "data-card active"; // 觸發發光邊框特效
            
            // 替換側邊欄的內容
            sidebar.innerHTML = `
                <h3 style="color: #4A90E2; border-bottom-color: #4A90E2;">📈 數據細節</h3>
                <p><b>📅 日期:</b> \${{selectedData.date}}</p>
                <div class="price-box">
                    <span>💰 QQQ 股價:</span>
                    <span style="color: #222;">\${{selectedData.close.toFixed(2)}}</span>
                </div>
                <h4 style="margin: 10px 0 5px 0; color: #555; font-size: 14px;">五線譜五個價位：</h4>
                <ul>
                    <li><span style="color:\${{colors[4]}}; font-weight:bold;">🔴 樂觀線:</span> <span>\${{selectedData.line5.toFixed(2)}}</span></li>
                    <li><span style="color:\${{colors[3]}}; font-weight:bold;">🟠 相對樂觀:</span> <span>\${{selectedData.line4.toFixed(2)}}</span></li>
                    <li><span style="color:\${{colors[2]}}; font-weight:bold;">🔵 趨勢中線:</span> <span>\${{selectedData.line3.toFixed(2)}}</span></li>
                    <li><span style="color:\${{colors[1]}}; font-weight:bold;">🟢 相對悲觀:</span> <span>\${{selectedData.line2.toFixed(2)}}</span></li>
                    <li><span style="color:\${{colors[0]}}; font-weight:bold;">🟢 悲觀線:</span> <span>\${{selectedData.line1.toFixed(2)}}</span></li>
                </ul>
            `;
        }}
    }});
</script>

</body>
</html>
"""

# 5. 寫入 index.html (GitHub Actions 隨後會自動將此檔案 Commit 提交)
with open("index.html", "w", encoding="utf-8") as f:
    f.write(html_template)

print("全新的互動式 index.html 已成功生成！")
