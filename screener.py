"""
日本株スクリーナー用データ作成スクリプト
- 東証の全上場銘柄(プライム/スタンダード/グロース)の日足を Yahoo Finance から取得
- スイング向けの指標とチャート分析を計算して site/ に出力
  site/data.json      … スクリーニング用(全銘柄)
  site/c/<コード>.json … チャート用(日足・週足・月足と分析結果)
GitHub Actions で平日の夕方に自動実行され、site/ が GitHub Pages に公開される。
"""
import json
import math
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import shutil

import numpy as np
import pandas as pd

import analysis

ROOT = Path(__file__).parent
TICKERS_CSV = ROOT / "tickers.csv"
SITE = ROOT / "site"
DATA_JSON = SITE / "data.json"
JPX_LIST_URL = (
    "https://www.jpx.co.jp/markets/statistics-equities/misc/"
    "tvdivq0000001vg2-att/data_j.xls"
)
JST = timezone(timedelta(hours=9))
BATCH = 100          # 1回にまとめて取得する銘柄数
HISTORY = "5y"       # 月足チャート用に5年分取る
KEEP = {"d": 370, "w": 260, "m": 60}  # チャートに載せる本数(日足約1年半・週足5年・月足5年)


# ---------------------------------------------------------------- 銘柄一覧
UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36",
    "Accept-Language": "ja,en;q=0.8",
}
JPX_PAGE_URL = "https://www.jpx.co.jp/markets/statistics-equities/misc/01.html"


def _download(url: str) -> bytes:
    import urllib.request

    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as res:
        return res.read()


def _jpx_xls() -> bytes:
    """JPXから一覧ファイルを取得。直接URLがダメなら掲載ページからリンクを探す。"""
    import re

    try:
        return _download(JPX_LIST_URL)
    except Exception as e:  # noqa: BLE001
        print(f"  直接取得に失敗: {e}")
    html = _download(JPX_PAGE_URL).decode("utf-8", "ignore")
    m = re.search(r'href="([^"]+\.xlsx?)"', html)
    if not m:
        raise RuntimeError("掲載ページに一覧ファイルのリンクが見つかりません")
    link = m.group(1)
    if link.startswith("/"):
        link = "https://www.jpx.co.jp" + link
    return _download(link)


def _col(df: pd.DataFrame, *words: str) -> str:
    """列名のゆれに対応: words を全部含む列名を返す。"""
    for c in df.columns:
        if all(w in str(c) for w in words):
            return c
    raise KeyError(f"列が見つかりません: {words} / 実際の列: {list(df.columns)}")


def _parse_jpx(df: pd.DataFrame) -> pd.DataFrame:
    df = df.astype(str)
    code, name = _col(df, "コード"), _col(df, "銘柄名")
    mkt = _col(df, "市場")
    try:
        sec = _col(df, "33業種", "区分")
    except KeyError:
        sec = _col(df, "業種")
    df = df[df[mkt].str.contains("内国株式|プライム|スタンダード|グロース", na=False)]
    df = df[~df[mkt].str.contains("ETF|REIT|外国|PRO", na=False)].copy()
    market = df[mkt].str.extract(r"(プライム|スタンダード|グロース)")[0]
    out = pd.DataFrame(
        {
            "code": df[code].str.strip().str.replace(r"\.0$", "", regex=True),
            "name": df[name].str.strip(),
            "market": market,
            "sector": df[sec].str.strip(),
        }
    )
    return out.dropna(subset=["market"])


def _read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        for enc in ("utf-8-sig", "cp932"):
            try:
                return pd.read_csv(path, dtype=str, encoding=enc)
            except UnicodeDecodeError:
                continue
    return pd.read_excel(path, dtype=str)


def _local_list() -> pd.DataFrame:
    """リポジトリに手動で置いた一覧ファイル(xls / xlsx / csv、tickers.csv以外)を探す。"""
    files = [
        p for p in ROOT.iterdir()
        if p.suffix.lower() in (".xls", ".xlsx", ".csv") and p.name != TICKERS_CSV.name
    ]
    if not files:
        raise FileNotFoundError("一覧ファイル(xls/xlsx/csv)がリポジトリにありません")
    errors = []
    for p in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            return _read_any(p)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{p.name}: {e}")
    raise RuntimeError(" / ".join(errors))


def load_tickers() -> pd.DataFrame:
    """銘柄一覧を ①リポジトリに置いた一覧ファイル ②JPXサイト ③前回の tickers.csv の順で取得。"""
    import io

    sources = [
        ("リポジトリの一覧ファイル", _local_list),
        ("JPXサイト", lambda: pd.read_excel(io.BytesIO(_jpx_xls()), dtype=str)),
    ]
    for label, fn in sources:
        try:
            out = _parse_jpx(fn())
            if len(out) < 1000:
                raise RuntimeError(f"銘柄数が少なすぎます({len(out)})")
            out.to_csv(TICKERS_CSV, index=False)
            print(f"銘柄一覧を取得({label}): {len(out)}銘柄")
            return out
        except Exception as e:  # noqa: BLE001
            print(f"銘柄一覧の取得に失敗({label}): {e}")
    if TICKERS_CSV.exists():
        print("前回保存した tickers.csv を使います。")
        return pd.read_csv(TICKERS_CSV, dtype=str)
    sys.exit(
        "銘柄一覧を取得できませんでした。JPXの上場銘柄一覧ファイル(xls/xlsx/csv)を"
        "リポジトリに置いてください。"
    )


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


def _rows(df: pd.DataFrame, n: int) -> list:
    df = df.tail(n)
    return [
        [int(i.strftime("%Y%m%d")), r(o, 1), r(h, 1), r(l, 1), r(c, 1), int(v)]
        for i, o, h, l, c, v in zip(df.index, df["Open"], df["High"], df["Low"], df["Close"], df["Volume"])
    ]


def build(tickers: pd.DataFrame, prices: dict[str, pd.DataFrame]) -> dict:
    cdir = SITE / "c"
    cdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for t in tickers.itertuples():
        d = prices.get(t.code)
        if d is None:
            continue
        m = calc(d.tail(300))
        if m is None:
            continue
        try:
            frames = {
                "d": d,
                "w": analysis.to_weekly_monthly(d, "W-FRI"),
                "m": analysis.to_weekly_monthly(d, "ME"),
            }
            ana = {tf: analysis.analyze(f.tail(analysis.PARAMS[tf]["look"] + 10), tf) for tf, f in frames.items()}
            flags = analysis.screen_flags(ana["d"], float(d["Close"].iloc[-1]))
            chart = {tf: _rows(f, KEEP[tf]) for tf, f in frames.items()}
            chart["a"] = ana
            (cdir / f"{t.code}.json").write_text(
                json.dumps(chart, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
            )
        except Exception as e:  # noqa: BLE001
            print(f"  {t.code} の分析でエラー: {e}")
            flags = {"sup": None, "tlu": None, "fib": False, "pat": ""}
        rows.append({"code": t.code, "name": t.name, "mkt": t.market, "sec": t.sector, **m, **flags})
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
    if SITE.exists():
        shutil.rmtree(SITE)
    SITE.mkdir()
    out = build(tickers, prices)
    for f in ("index.html", "icon.png"):
        shutil.copy(ROOT / f, SITE / f)
    DATA_JSON.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"data.json を保存: {out['date']} / {out['count']}銘柄")


if __name__ == "__main__":
    main()
