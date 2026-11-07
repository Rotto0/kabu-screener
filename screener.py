"""
日本株スクリーナー用データ作成スクリプト
- 東証の全上場銘柄(プライム/スタンダード/グロース)の日足を Yahoo Finance から取得
- スイング向けの指標を計算して data.json に保存
GitHub Actions で平日の夕方に自動実行される想定。
"""
import json
import math
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
TICKERS_CSV = ROOT / "tickers.csv"
DATA_JSON = ROOT / "data.json"
JPX_LIST_URL = (
    "https://www.jpx.co.jp/markets/statistics-equities/misc/"
    "tvdivq0000001vg2-att/data_j.xls"
)
JST = timezone(timedelta(hours=9))
BATCH = 150          # 1回にまとめて取得する銘柄数
HISTORY = "14mo"     # 52週高値を出すために1年強の履歴を取る


# ---------------------------------------------------------------- 銘柄一覧
def load_tickers() -> pd.DataFrame:
    """JPX公開の上場銘柄一覧を取得。失敗したら前回保存した tickers.csv を使う。"""
    try:
        df = pd.read_excel(JPX_LIST_URL, dtype=str)
        df = df[df["市場・商品区分"].str.contains("内国株式", na=False)]
        df["market"] = (
            df["市場・商品区分"]
            .str.replace("（内国株式）", "", regex=False)
            .str.replace("(内国株式)", "", regex=False)
        )
        out = pd.DataFrame(
            {
                "code": df["コード"].str.strip(),
                "name": df["銘柄名"].str.strip(),
                "market": df["market"].str.strip(),
                "sector": df["33業種区分"].str.strip(),
            }
        )
        out.to_csv(TICKERS_CSV, index=False)
        print(f"JPX一覧を取得: {len(out)}銘柄")
        return out
    except Exception as e:  # noqa: BLE001
        print(f"JPX一覧の取得に失敗({e})。保存済みの一覧を使います。")
        return pd.read_csv(TICKERS_CSV, dtype=str)


# ---------------------------------------------------------------- 株価取得
def fetch_prices(codes: list[str]) -> dict[str, pd.DataFrame]:
    import yfinance as yf

    result: dict[str, pd.DataFrame] = {}
    for i in range(0, len(codes), BATCH):
        chunk = codes[i : i + BATCH]
        symbols = [f"{c}.T" for c in chunk]
        for attempt in range(3):
            try:
                raw = yf.download(
                    symbols,
                    period=HISTORY,
                    interval="1d",
                    group_by="ticker",
                    auto_adjust=False,
                    threads=True,
                    progress=False,
                )
                break
            except Exception as e:  # noqa: BLE001
                print(f"  取得エラー(再試行 {attempt + 1}/3): {e}")
                time.sleep(10)
        else:
            continue
        for c, s in zip(chunk, symbols):
            try:
                d = raw[s] if isinstance(raw.columns, pd.MultiIndex) else raw
                d = d[["Open", "High", "Low", "Close", "Volume"]].dropna()
                if len(d) >= 30:
                    result[c] = d
            except KeyError:
                pass
        print(f"  {min(i + BATCH, len(codes))}/{len(codes)} 取得済み")
        time.sleep(2)
    return result


# ---------------------------------------------------------------- 指標計算
def rsi(close: pd.Series, n: int = 14) -> float:
    diff = close.diff()
    up = diff.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-diff.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    last_down = down.iloc[-1]
    if last_down == 0:
        return 100.0
    return float(100 - 100 / (1 + up.iloc[-1] / last_down))


def r(x, nd=2):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return None
    return round(float(x), nd)


def calc(d: pd.DataFrame) -> dict | None:
    c, h, v = d["Close"], d["High"], d["Volume"]
    if len(c) < 30 or c.iloc[-1] <= 0:
        return None
    last, prev = c.iloc[-1], c.iloc[-2]
    ma = lambda n: c.rolling(n).mean().iloc[-1] if len(c) >= n else np.nan  # noqa: E731
    ma5, ma25, ma75 = ma(5), ma(25), ma(75)
    ma25_5ago = c.rolling(25).mean().iloc[-6] if len(c) >= 30 else np.nan
    vol_avg20 = v.iloc[-21:-1].mean()  # 当日を除く直近20日平均
    high20_prev = h.iloc[-21:-1].max()
    high52_prev = h.iloc[-251:-1].max()
    high52 = h.iloc[-250:].max()
    return {
        "price": r(last, 1),
        "chg": r((last / prev - 1) * 100),
        "vol": int(v.iloc[-1]),
        "vr": r(v.iloc[-1] / vol_avg20) if vol_avg20 > 0 else None,
        "tv": r(last * v.iloc[-1] / 1e8, 2),                 # 売買代金(億円)
        "tv20": r((c.iloc[-21:-1] * v.iloc[-21:-1]).mean() / 1e8, 2),
        "d25": r((last / ma25 - 1) * 100) if ma25 == ma25 else None,   # 25日線乖離
        "d75": r((last / ma75 - 1) * 100) if ma75 == ma75 else None,
        "a5": bool(last > ma5) if ma5 == ma5 else None,
        "up25": bool(ma25 > ma25_5ago) if ma25 == ma25 and ma25_5ago == ma25_5ago else None,
        "po": bool(ma5 > ma25 > ma75) if ma75 == ma75 else None,       # パーフェクトオーダー
        "h20": bool(last > high20_prev),                               # 20日高値更新(終値)
        "h52": bool(last > high52_prev) if len(h) >= 120 else None,    # 52週高値更新(終値)
        "f52": r((last / high52 - 1) * 100),                           # 52週高値からの距離
        "rsi": r(rsi(c), 1),
        "date": d.index[-1].strftime("%Y-%m-%d"),
    }


def build(tickers: pd.DataFrame, prices: dict[str, pd.DataFrame]) -> dict:
    rows = []
    for t in tickers.itertuples():
        d = prices.get(t.code)
        if d is None:
            continue
        m = calc(d)
        if m is None:
            continue
        rows.append({"code": t.code, "name": t.name, "mkt": t.market, "sec": t.sector, **m})
    # 一番多い日付を「データ日付」とし、それより古い銘柄(売買停止など)は除外
    latest = pd.Series([x["date"] for x in rows]).mode().iloc[0] if rows else ""
    rows = [x for x in rows if x["date"] == latest]
    for x in rows:
        del x["date"]
    keys = list(rows[0].keys()) if rows else []
    return {
        "date": latest,
        "updated": datetime.now(JST).strftime("%Y-%m-%d %H:%M"),
        "count": len(rows),
        "keys": keys,
        "rows": [[x[k] for k in keys] for x in rows],
    }


def main():
    tickers = load_tickers()
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None  # テスト用: 先頭N銘柄だけ
    if limit:
        tickers = tickers.head(limit)
    prices = fetch_prices(tickers["code"].tolist())
    print(f"株価を取得できた銘柄: {len(prices)}")
    if len(prices) < len(tickers) * 0.5:
        sys.exit("取得できた銘柄が半分未満のため、data.json を更新せずに終了します。")
    out = build(tickers, prices)
    DATA_JSON.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"data.json を保存: {out['date']} / {out['count']}銘柄")


if __name__ == "__main__":
    main()
