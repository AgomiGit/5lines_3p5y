import json

import fear_greed
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression

from market_signals import (
    fear_greed_band,
    forward_pe_band,
    get_forward_pe_data,
    get_vix_data,
    vix_band,
)


TICKERS = [
    "QQQ",
    "00662.TW",
    "00830.TW",
    "VOO",
    "2330.TW",
    "00757.TW",
    "009815.TWO",
    "SMH",
]
DAYS_WINDOW = 875


def build_ticker_data(ticker):
    print(f"正在抓取 {ticker} 的歷史數據...")
    source = yf.download(ticker, start="2022-01-01", auto_adjust=False, progress=False)
    if source.empty:
        raise RuntimeError(f"找不到 {ticker} 的歷史數據")

    if isinstance(source.columns, pd.MultiIndex):
        source.columns = source.columns.droplevel(1)

    source.index = pd.to_datetime(source.index)
    frame = (
        source[["Close"]]
        .rename(columns={"Close": "close"})
        .dropna(subset=["close"])
        .tail(DAYS_WINDOW)
        .copy()
    )
    frame["X"] = np.arange(len(frame))

    model = LinearRegression()
    model.fit(frame[["X"]], frame["close"])
    frame["TL"] = model.predict(frame[["X"]])

    standard_deviation = np.std(frame["close"].values - frame["TL"].values)
    frame["TL_plus_4SD"] = frame["TL"] + (4 * standard_deviation)
    frame["TL_plus_3SD"] = frame["TL"] + (3 * standard_deviation)
    frame["TL_plus_2SD"] = frame["TL"] + (2 * standard_deviation)
    frame["TL_plus_1SD"] = frame["TL"] + standard_deviation
    frame["TL_minus_1SD"] = frame["TL"] - standard_deviation
    frame["TL_minus_2SD"] = frame["TL"] - (2 * standard_deviation)
    frame["TL_minus_3SD"] = frame["TL"] - (3 * standard_deviation)
    frame["TL_minus_4SD"] = frame["TL"] - (4 * standard_deviation)

    chart_data = []
    for date_index, row in frame.iterrows():
        chart_data.append(
            {
                "date": date_index.strftime("%Y-%m-%d"),
                "close": round(row["close"], 2),
                "tl": round(row["TL"], 2),
                "p4sd": round(row["TL_plus_4SD"], 2),
                "p3sd": round(row["TL_plus_3SD"], 2),
                "p1sd": round(row["TL_plus_1SD"], 2),
                "p2sd": round(row["TL_plus_2SD"], 2),
                "m1sd": round(row["TL_minus_1SD"], 2),
                "m2sd": round(row["TL_minus_2SD"], 2),
                "m3sd": round(row["TL_minus_3SD"], 2),
                "m4sd": round(row["TL_minus_4SD"], 2),
            }
        )

    is_taiwan_stock = ticker.endswith((".TW", ".TWO"))
    return {
        "currencySymbol": "NT$" if is_taiwan_stock else "$",
        "lastDate": chart_data[-1]["date"],
        "sentimentDescription": (
            "統計學價格軌道與 CNN 美股市場情緒參考"
            if is_taiwan_stock
            else "統計學價格軌道與美股即時情緒指標"
        ),
        "chartData": chart_data,
    }


def get_fear_greed():
    print("正在獲取 CNN Fear & Greed 當前分數...")
    try:
        data = fear_greed.get()
        return float(data["score"]), data["rating"].upper()
    except Exception as error:
        print(f"無法取得 CNN Fear & Greed：{error}")
        return 50.0, "資料暫缺"


def gauge_color(score):
    if score <= 25:
        return "#cc0000"
    if score <= 45:
        return "#ff9900"
    if score <= 55:
        return "#888888"
    if score <= 75:
        return "#66cc00"
    return "#008800"


dashboard_data = {ticker: build_ticker_data(ticker) for ticker in TICKERS}
current_fg_score, current_fg_rating = get_fear_greed()
fear_greed_available = current_fg_rating != "資料暫缺"
gauge_score = current_fg_score if fear_greed_available else 0.0
current_gauge_color = gauge_color(current_fg_score) if fear_greed_available else "#8a939d"
current_fg_label, current_fg_tone = fear_greed_band(
    current_fg_score if fear_greed_available else None
)
vix_data = get_vix_data()
vix_label, vix_tone = vix_band(vix_data["value"])
forward_pe_data = get_forward_pe_data()
forward_pe_label, forward_pe_tone = forward_pe_band(forward_pe_data["value"])

ticker_options = "\n".join(
    f'<option value="{ticker}">{ticker}</option>' for ticker in TICKERS
)

html = """<!DOCTYPE html>
<html lang="zh-Hant">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>樂活五線譜投資決策儀表板</title>
    <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 0; padding: 15px; background-color: #f8f9fa; color: #333; display: flex; flex-direction: column; align-items: center; }
        h1 { color: #111; font-size: 24px; margin: 10px 0 5px; text-align: center; }
        h2 { color: #6c757d; font-size: 13px; font-weight: normal; margin: 0 0 20px; text-align: center; padding: 0 10px; }
        .ticker-picker { width: 100%; max-width: 1200px; margin-bottom: 20px; text-align: center; }
        .ticker-picker label { display: block; color: #555; font-size: 13px; font-weight: 600; margin-bottom: 7px; }
        .ticker-picker select { min-width: 220px; padding: 10px 38px 10px 14px; border: 1px solid #cbd5e1; border-radius: 8px; background: #fff; color: #111; font-size: 16px; }
        .signal-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; width: 100%; max-width: 1200px; margin-bottom: 20px; }
        .signal-card { min-height: 190px; padding: 24px 28px; background: #f4f9fa; border: 2px solid #e2e7ec; border-radius: 20px; box-shadow: 0 3px 10px rgba(15, 23, 42, 0.04); box-sizing: border-box; text-align: center; display: flex; flex-direction: column; justify-content: center; align-items: center; gap: 12px; }
        .signal-title { color: #667085; font-size: 1.15rem; font-weight: 500; line-height: 1.35; }
        .signal-title a { color: #2563c7; font-weight: 650; text-decoration: none; }
        .signal-title a:hover, .signal-title a:focus-visible { text-decoration: underline; }
        .signal-value { color: #1f2937; font-size: clamp(2rem, 4vw, 2.75rem); font-weight: 750; line-height: 1; font-variant-numeric: tabular-nums; }
        .signal-pill { max-width: 100%; padding: 7px 22px; border: 2px solid currentColor; border-radius: 999px; font-size: 1.05rem; font-weight: 700; line-height: 1.2; box-sizing: border-box; }
        .signal-meta { color: #667085; font-size: 0.95rem; font-variant-numeric: tabular-nums; }
        .tone-danger .signal-value, .signal-pill.tone-danger { color: #c61d1d; }
        .tone-warning .signal-value, .signal-pill.tone-warning { color: #d65f0b; }
        .tone-positive .signal-value, .signal-pill.tone-positive { color: #16834b; }
        .tone-neutral .signal-value, .signal-pill.tone-neutral { color: #2563c7; }
        .tone-muted .signal-value, .signal-pill.tone-muted { color: #7b8492; }
        .gauge-wrapper { background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); width: 100%; max-width: 1200px; margin-bottom: 20px; box-sizing: border-box; }
        .gauge-title { font-size: 14px; font-weight: bold; margin-bottom: 12px; color: #111; }
        .gauge-bar-bg { background: #e9ecef; height: 20px; border-radius: 10px; position: relative; overflow: hidden; display: flex; }
        .gauge-fill { background: __GAUGE_COLOR__; width: __FG_SCORE__%; height: 100%; border-radius: 10px 0 0 10px; }
        .gauge-text { position: absolute; left: calc(__FG_SCORE__% - 25px); top: 1px; color: __GAUGE_TEXT_COLOR__; font-weight: bold; font-size: 12px; text-shadow: __GAUGE_TEXT_SHADOW__; }
        .gauge-labels { display: flex; justify-content: space-between; margin-top: 6px; font-size: 11px; color: #6c757d; font-weight: bold; }
        .main-content { display: flex; flex-direction: column; width: 100%; max-width: 1200px; gap: 20px; margin-bottom: 20px; }
        .chart-container { background: white; padding: 15px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); min-width: 0; }
        .chart-title { font-size: 15px; font-weight: bold; text-align: left; margin-bottom: 10px; color: #111; }
        .data-card { background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border: 2px solid #e0e6ed; box-sizing: border-box; width: 100%; text-align: left; }
        .sidebar-title { margin-top: 0; color: #4A90E2; border-bottom: 2px solid #4A90E2; padding-bottom: 8px; font-size: 18px; }
        .price-box { background-color: #f1f5f9; padding: 12px 20px; border-radius: 6px; font-size: 20px; font-weight: bold; margin: 15px 0; display: flex; justify-content: space-between; align-items: center; }
        .grid-list { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; padding: 0; margin: 0; list-style: none; }
        .grid-list li { padding: 12px; background: #f8f9fa; border-radius: 6px; border-left: 4px solid #ccc; display: flex; flex-direction: column; gap: 4px; }
        .line-name { font-size: 12px; color: #666; font-weight: 500; }
        .line-val { font-size: 16px; font-weight: bold; color: #111; }
        .hoverlayer { display: none !important; visibility: hidden !important; opacity: 0 !important; }
        .info { margin-top: 20px; font-size: 11px; color: #8a939d; text-align: center; padding: 0 10px; }
        @media (max-width: 600px) {
            body { padding: 12px; }
            .signal-grid { grid-template-columns: 1fr; gap: 14px; }
            .signal-card { min-height: 165px; padding: 20px 16px; border-radius: 16px; }
            .signal-title { font-size: 1rem; }
            .signal-pill { font-size: 0.95rem; padding: 7px 14px; }
            .signal-meta { font-size: 0.875rem; }
            .gauge-labels { font-size: 9px; }
            .price-box { font-size: 17px; }
        }
    </style>
</head>
<body>
    <h1 id="page-title"></h1>
    <h2 id="page-description"></h2>

    <div class="ticker-picker">
        <label for="ticker-select">選擇商品</label>
        <select id="ticker-select" aria-label="選擇商品">
__TICKER_OPTIONS__
        </select>
    </div>

    <div class="signal-grid" aria-label="市場關鍵指標">
        <article id="price-card" class="signal-card tone-neutral">
            <div id="close-title" class="signal-title"></div>
            <div id="close-value" class="signal-value"></div>
            <div id="price-signal" class="signal-pill tone-neutral"></div>
        </article>
        <article class="signal-card __FG_TONE__">
            <div class="signal-title">CNN 即時情緒狀態</div>
            <div class="signal-value">__FG_DISPLAY__</div>
            <div class="signal-pill __FG_TONE__">__FG_LABEL__</div>
        </article>
        <article class="signal-card __VIX_TONE__">
            <div class="signal-title">CBOE VIX 波動指數
                （<a href="https://finance.yahoo.com/quote/%5EVIX/" target="_blank" rel="noopener noreferrer">Yahoo</a> ·
                <a href="https://www.cboe.com/tradable_products/vix/" target="_blank" rel="noopener noreferrer">CBOE</a>）
            </div>
            <div class="signal-value">__VIX_DISPLAY__</div>
            <div class="signal-pill __VIX_TONE__">__VIX_LABEL__</div>
            <div class="signal-meta">__VIX_META__</div>
        </article>
        <article class="signal-card __PE_TONE__">
            <div class="signal-title">Nasdaq 100 Forward P/E
                （<a href="https://dollarliquidity.com/en/valuation/nasdaq-100-forward-pe" target="_blank" rel="noopener noreferrer">Dollar Liquidity</a>）
            </div>
            <div class="signal-value">__PE_DISPLAY__</div>
            <div class="signal-pill __PE_TONE__">__PE_LABEL__</div>
            <div class="signal-meta">__PE_META__</div>
        </article>
    </div>

    <div class="gauge-wrapper">
        <div class="gauge-title">📊 CNN Fear &amp; Greed Index 即時量表（當前分數：__FG_DISPLAY__）</div>
        <div class="gauge-bar-bg">
            <div class="gauge-fill"></div>
            <span class="gauge-text">__FG_DISPLAY__</span>
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
            <div id="chart-title" class="chart-title"></div>
            <div id="plotly-chart"></div>
        </div>
        <div id="sidebar-content" class="data-card"></div>
    </div>

    <div class="info">
        <p>最後更新日期：<span id="last-date"></span>｜本網頁由 GitHub Actions 每日自動更新</p>
        <p>VIX：Yahoo Finance／CBOE｜Forward P/E：<a href="https://dollarliquidity.com/en/valuation/nasdaq-100-forward-pe" target="_blank" rel="noopener noreferrer">Dollar Liquidity</a>｜資料僅供參考，不構成投資建議</p>
    </div>

    <script>
        const dashboards = __DASHBOARD_DATA__;
        const tickerSelect = document.getElementById('ticker-select');
        const chartDiv = document.getElementById('plotly-chart');
        let selectedTicker = tickerSelect.value;

        function formatPrice(value, currencySymbol) {
            return currencySymbol + Number(value).toFixed(2);
        }

        function getPriceSignal(row, currencySymbol) {
            if (row.close <= row.m2sd) return {label: `極悲觀區（≤ ${formatPrice(row.m2sd, currencySymbol)}）`, tone: 'tone-danger'};
            if (row.close <= row.m1sd) return {label: `相對悲觀區（≤ ${formatPrice(row.m1sd, currencySymbol)}）`, tone: 'tone-warning'};
            if (row.close < row.p1sd) return {label: '合理價格區（-1SD～+1SD）', tone: 'tone-neutral'};
            if (row.close < row.p2sd) return {label: `相對樂觀區（≥ ${formatPrice(row.p1sd, currencySymbol)}）`, tone: 'tone-warning'};
            return {label: `極樂觀區（≥ ${formatPrice(row.p2sd, currencySymbol)}）`, tone: 'tone-danger'};
        }

        function renderSidebar(row) {
            const dashboard = dashboards[selectedTicker];
            const currency = dashboard.currencySymbol;
            document.getElementById('sidebar-content').innerHTML = `
                <h3 class="sidebar-title">📈 歷史數據細節（📅 ${row.date}）</h3>
                <div class="price-box">
                    <span>💰 ${selectedTicker} 當日股價：</span>
                    <span style="color: #222;">${formatPrice(row.close, currency)}</span>
                </div>
                <ul class="grid-list">
                    <li style="border-left-color: #7f1d1d;"><span class="line-name">+4SD 線</span><span class="line-val">${formatPrice(row.p4sd, currency)}</span></li>
                    <li style="border-left-color: #c2410c;"><span class="line-name">+3SD 線</span><span class="line-val">${formatPrice(row.p3sd, currency)}</span></li>
                    <li style="border-left-color: #dc3545;"><span class="line-name">🔴 極樂觀線 (+2SD)</span><span class="line-val">${formatPrice(row.p2sd, currency)}</span></li>
                    <li style="border-left-color: #ffc107;"><span class="line-name">🟠 相對樂觀 (+1SD)</span><span class="line-val">${formatPrice(row.p1sd, currency)}</span></li>
                    <li style="border-left-color: #28a745;"><span class="line-name">🔵 趨勢中軸 (TL)</span><span class="line-val">${formatPrice(row.tl, currency)}</span></li>
                    <li style="border-left-color: #007bff;"><span class="line-name">🟢 相對悲觀 (-1SD)</span><span class="line-val">${formatPrice(row.m1sd, currency)}</span></li>
                    <li style="border-left-color: #6f42c1;"><span class="line-name">🟣 極悲觀線 (-2SD)</span><span class="line-val">${formatPrice(row.m2sd, currency)}</span></li>
                    <li style="border-left-color: #4338ca;"><span class="line-name">-3SD 線</span><span class="line-val">${formatPrice(row.m3sd, currency)}</span></li>
                    <li style="border-left-color: #1e3a8a;"><span class="line-name">-4SD 線</span><span class="line-val">${formatPrice(row.m4sd, currency)}</span></li>
                </ul>`;
        }

        function renderDashboard(ticker) {
            selectedTicker = ticker;
            const dashboard = dashboards[ticker];
            const stockData = dashboard.chartData;
            const latest = stockData[stockData.length - 1];
            const dates = stockData.map(row => row.date);

            document.title = `${ticker} 投資決策儀表板`;
            document.getElementById('page-title').textContent = `${ticker} 🎯 投資決策儀表板`;
            document.getElementById('page-description').textContent = `數據驅動看板：${dashboard.sentimentDescription}`;
            document.getElementById('close-title').textContent = `${ticker} 最新股價`;
            document.getElementById('close-value').textContent = formatPrice(latest.close, dashboard.currencySymbol);
            const priceSignal = getPriceSignal(latest, dashboard.currencySymbol);
            const priceCard = document.getElementById('price-card');
            const pricePill = document.getElementById('price-signal');
            ['tone-danger', 'tone-warning', 'tone-neutral', 'tone-positive', 'tone-muted'].forEach(tone => {
                priceCard.classList.remove(tone);
                pricePill.classList.remove(tone);
            });
            priceCard.classList.add(priceSignal.tone);
            pricePill.classList.add(priceSignal.tone);
            pricePill.textContent = priceSignal.label;
            document.getElementById('chart-title').textContent = `📈 ${ticker} 樂活五線譜趨勢圖`;
            document.getElementById('last-date').textContent = dashboard.lastDate;
            renderSidebar(latest);

            const traces = [
                { x: dates, y: stockData.map(row => row.close), name: `${ticker} 收盤價`, type: 'scatter', mode: 'lines', line: {color: '#000000', width: 2} },
                { x: dates, y: stockData.map(row => row.p4sd), name: '+4SD 線', type: 'scatter', line: {color: '#7f1d1d', width: 1.25, dash: 'dot'} },
                { x: dates, y: stockData.map(row => row.p3sd), name: '+3SD 線', type: 'scatter', line: {color: '#c2410c', width: 1.25, dash: 'dashdot'} },
                { x: dates, y: stockData.map(row => row.p2sd), name: '極樂觀線 (+2SD)', type: 'scatter', line: {color: '#dc3545', width: 1.5, dash: 'dash'} },
                { x: dates, y: stockData.map(row => row.p1sd), name: '相對樂觀線 (+1SD)', type: 'scatter', line: {color: '#ffc107', width: 1.5, dash: 'dash'} },
                { x: dates, y: stockData.map(row => row.tl), name: '趨勢中軸線 (TL)', type: 'scatter', line: {color: '#28a745', width: 2} },
                { x: dates, y: stockData.map(row => row.m1sd), name: '相對悲觀線 (-1SD)', type: 'scatter', line: {color: '#007bff', width: 1.5, dash: 'dash'} },
                { x: dates, y: stockData.map(row => row.m2sd), name: '極悲觀線 (-2SD)', type: 'scatter', line: {color: '#6f42c1', width: 1.5, dash: 'dash'} },
                { x: dates, y: stockData.map(row => row.m3sd), name: '-3SD 線', type: 'scatter', line: {color: '#4338ca', width: 1.25, dash: 'dashdot'} },
                { x: dates, y: stockData.map(row => row.m4sd), name: '-4SD 線', type: 'scatter', line: {color: '#1e3a8a', width: 1.25, dash: 'dot'} }
            ];
            const layout = {
                hovermode: 'x unified',
                margin: {r: 5, l: 45, t: 10, b: 30},
                height: 400,
                legend: {orientation: 'h', y: -0.2, x: 0.5, xanchor: 'center'},
                xaxis: {type: 'date', gridcolor: '#f0f0f0', fixedrange: true},
                yaxis: {gridcolor: '#f0f0f0', fixedrange: true},
                plot_bgcolor: '#ffffff',
                paper_bgcolor: '#ffffff'
            };
            Plotly.react(chartDiv, traces, layout, {scrollZoom: false, displayModeBar: false, responsive: true});
        }

        renderDashboard(selectedTicker);
        tickerSelect.addEventListener('change', event => renderDashboard(event.target.value));
        chartDiv.on('plotly_click', event => {
            const row = dashboards[selectedTicker].chartData[event.points[0].pointIndex];
            if (row) renderSidebar(row);
        });
    </script>
</body>
</html>
"""

replacements = {
    "__DASHBOARD_DATA__": json.dumps(dashboard_data, ensure_ascii=False),
    "__TICKER_OPTIONS__": ticker_options,
    "__FG_SCORE__": str(round(gauge_score, 1)),
    "__FG_DISPLAY__": (
        str(round(current_fg_score, 1)) if fear_greed_available else "—"
    ),
    "__FG_LABEL__": current_fg_label,
    "__FG_TONE__": current_fg_tone,
    "__VIX_DISPLAY__": (
        f'{vix_data["value"]:.2f}' if vix_data["value"] is not None else "—"
    ),
    "__VIX_LABEL__": vix_label,
    "__VIX_TONE__": vix_tone,
    "__VIX_META__": (
        f'{vix_data["date"]}｜前收 {vix_data["previous"]:.2f}'
        if vix_data["value"] is not None
        else "最新資料暫缺"
    ),
    "__PE_DISPLAY__": (
        f'{forward_pe_data["value"]:.2f}x ({forward_pe_data["change"]:+.2f})'
        if forward_pe_data["value"] is not None
        else "—"
    ),
    "__PE_LABEL__": forward_pe_label,
    "__PE_TONE__": forward_pe_tone,
    "__PE_META__": (
        f'{forward_pe_data["date"]}｜前值 {forward_pe_data["previous"]:.2f}x'
        if forward_pe_data["value"] is not None
        else "最新資料暫缺"
    ),
    "__GAUGE_COLOR__": current_gauge_color,
    "__GAUGE_TEXT_COLOR__": "#333" if gauge_score < 10 else "#fff",
    "__GAUGE_TEXT_SHADOW__": (
        "none" if gauge_score < 10 else "1px 1px 2px rgba(0,0,0,0.5)"
    ),
}
for placeholder, value in replacements.items():
    html = html.replace(placeholder, value)

with open("index.html", "w", encoding="utf-8") as output_file:
    output_file.write(html)

print(f"已產生包含 {len(dashboard_data)} 個商品的 index.html")
