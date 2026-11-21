import re
from datetime import datetime
from html.parser import HTMLParser
from urllib.request import Request, urlopen


FORWARD_PE_URL = "https://dollarliquidity.com/en/valuation"


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        text = data.strip()
        if text:
            self.parts.append(text)


def fear_greed_band(score):
    if score is None:
        return "資料暫缺", "tone-muted"
    if score < 25:
        return "極度恐慌（0～<25）", "tone-danger"
    if score < 45:
        return "恐慌（25～<45）", "tone-warning"
    if score <= 55:
        return "中立（45～55）", "tone-neutral"
    if score <= 75:
        return "貪婪（>55～75）", "tone-positive"
    return "極度貪婪（>75～100）", "tone-positive"


def vix_band(value):
    if value is None:
        return "資料暫缺", "tone-muted"
    if value < 15:
        return "市場平靜（<15）", "tone-positive"
    if value < 20:
        return "波動正常（15～<20）", "tone-neutral"
    if value < 30:
        return "市場焦慮（20～<30）", "tone-warning"
    if value < 40:
        return "市場恐慌（30～<40）", "tone-warning"
    return "極度恐慌（≥40）", "tone-danger"


def forward_pe_band(value):
    if value is None:
        return "資料暫缺", "tone-muted"
    if value < 18:
        return "非常便宜（低於 18 倍）", "tone-danger"
    if value < 22:
        return "相對便宜（18～<22 倍）", "tone-positive"
    if value < 26:
        return "估值合理（22～<26 倍）", "tone-neutral"
    if value < 30:
        return "估值偏高（26～<30 倍）", "tone-warning"
    return "估值昂貴（30 倍以上）", "tone-danger"


def get_vix_data():
    print("正在獲取 CBOE VIX 波動指數...")
    try:
        import pandas as pd
        import yfinance as yf

        source = yf.download(
            "^VIX", period="10d", interval="1d", auto_adjust=False, progress=False
        )
        if source.empty:
            raise RuntimeError("VIX 沒有可用資料")
        if isinstance(source.columns, pd.MultiIndex):
            source.columns = source.columns.droplevel(1)

        closes = source["Close"].dropna()
        if len(closes) < 2:
            raise RuntimeError("VIX 資料不足兩個交易日")

        return {
            "value": round(float(closes.iloc[-1]), 2),
            "previous": round(float(closes.iloc[-2]), 2),
            "date": pd.Timestamp(closes.index[-1]).strftime("%Y-%m-%d"),
        }
    except Exception as error:
        print(f"無法取得 VIX：{error}")
        return {"value": None, "previous": None, "date": None}


def parse_forward_pe_html(content):
    parser = _TextExtractor()
    parser.feed(content)
    text = re.sub(r"\s+", " ", " ".join(parser.parts))

    match = re.search(
        r"Nasdaq-100 Forward P/E\s+(\d+(?:\.\d+)?)\s*×.*?"
        r"As of\s+([A-Z][a-z]{2}\s+\d{1,2},\s+\d{4})\s*([+−-]\d+(?:\.\d+)?)",
        text,
    )
    if not match:
        raise ValueError("找不到 Nasdaq-100 Forward P/E 欄位")

    value = float(match.group(1))
    change = float(match.group(3).replace("−", "-"))
    data_date = datetime.strptime(match.group(2), "%b %d, %Y").strftime("%Y-%m-%d")
    return {
        "value": round(value, 2),
        "change": round(change, 2),
        "previous": round(value - change, 2),
        "date": data_date,
    }


def get_forward_pe_data():
    print("正在獲取 Nasdaq-100 Forward P/E...")
    try:
        request = Request(
            FORWARD_PE_URL,
            headers={
                "Accept": "text/html",
                "User-Agent": "5lines-3p5y-dashboard/1.0",
            },
        )
        with urlopen(request, timeout=20) as response:
            content = response.read().decode("utf-8")
        return parse_forward_pe_html(content)
    except Exception as error:
        print(f"無法取得 Nasdaq-100 Forward P/E：{error}")
        return {"value": None, "change": None, "previous": None, "date": None}
