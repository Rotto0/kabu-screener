"""
チャート分析(サポート・レジスタンス、トレンドライン、フィボナッチ、パターン判定)
日足・週足・月足どれにも同じ関数を使う。結果はあくまで機械的な目安。
"""
import numpy as np
import pandas as pd

# 足の種類ごとの設定: 山谷判定の幅, 価格の許容誤差, 見る期間(本数)
PARAMS = {
    "d": {"k": 5, "tol": 0.015, "look": 250, "fib": 120, "pat": 80},
    "w": {"k": 3, "tol": 0.03, "look": 156, "fib": 104, "pat": 60},
    "m": {"k": 2, "tol": 0.05, "look": 120, "fib": 60, "pat": 40},
}
FIB_RATIOS = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]


def to_weekly_monthly(d: pd.DataFrame, rule: str) -> pd.DataFrame:
    """日足から週足/月足を作る。日付は期間内の最後の取引日。"""
    x = d.copy()
    x["dt"] = x.index
    g = x.resample(rule).agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum", "dt": "last"}
    ).dropna()
    return g.set_index("dt")


def pivots(h: np.ndarray, l: np.ndarray, k: int):
    """左右k本の中で一番高い(安い)足を山(谷)とする。直近k本は未確定なので除く。"""
    highs, lows = [], []
    for i in range(k, len(h) - k):
        if h[i] == h[i - k : i + k + 1].max():
            highs.append(i)
        if l[i] == l[i - k : i + k + 1].min():
            lows.append(i)
    return highs, lows


def _sr_levels(prices: list[float], tol: float):
    """近い価格の山谷をまとめて水平線にする。"""
    levels = []
    for p in sorted(prices):
        if levels and p <= levels[-1]["sum"] / levels[-1]["n"] * (1 + tol):
            levels[-1]["sum"] += p
            levels[-1]["n"] += 1
        else:
            levels.append({"sum": p, "n": 1})
    return [{"p": lv["sum"] / lv["n"], "n": lv["n"]} for lv in levels if lv["n"] >= 2]


def _trendline(idx: list[int], lows: np.ndarray, highs: np.ndarray, closes: np.ndarray,
               up: bool, tol: float, last: int):
    """
    トレンドラインを探す。
    - 候補: 直近の谷(上昇ライン)または山(下降ライン)のうち2点を結んだ線
    - 条件: 2点目以降、終値がラインを tol 以上割り込んで(超えて)いない
            (直近3本は除外して判定し、割れ/ブレイクは別に表示する)
    - 採点: 線に触れた山谷の数が多いほど、期間が長いほど、最後の接触が新しいほど高得点
    """
    vals = lows if up else highs
    cand = idx[-14:]
    n_bars = last + 1
    best, best_score = None, -1.0
    for j in range(1, len(cand)):
        b = cand[j]
        for i in range(j):
            a = cand[i]
            if b - a < 4:
                continue
            ya, yb = vals[a], vals[b]
            if up and yb <= ya:
                continue
            if not up and yb >= ya:
                continue
            slope = (yb - ya) / (b - a)
            line = lambda x: ya + slope * (x - a)  # noqa: E731
            # 1点目〜2点目の間で、ヒゲが線を大きく突き抜けていないこと
            mid = np.arange(a, b + 1)
            if up and (vals[a : b + 1] < line(mid) * (1 - tol)).any():
                continue
            if not up and (vals[a : b + 1] > line(mid) * (1 + tol)).any():
                continue
            # 2点目以降、直近3本を除き終値が線を割っていない(超えていない)こと
            chk_end = max(b, last - 3)
            xs = np.arange(b, chk_end + 1)
            seg = closes[b : chk_end + 1]
            if up and (seg < line(xs) * (1 - tol)).any():
                continue
            if not up and (seg > line(xs) * (1 + tol)).any():
                continue
            # 接触回数(線の近くにある山谷の数)
            touch_idx = [k for k in idx if k >= a and abs(vals[k] - line(k)) / line(k) <= tol * 0.6]
            touches = len(touch_idx)
            span = (b - a) / n_bars
            recency = 1 - (last - max(touch_idx)) / n_bars
            score = touches * 2 + span * 3 + recency * 2
            if score > best_score:
                end = line(last)
                status = "on"
                if up and closes[-1] < end * (1 - tol * 0.3):
                    status = "break"
                if not up and closes[-1] > end * (1 + tol * 0.3):
                    status = "break"
                best_score = score
                best = {"a": int(a), "ya": float(ya), "end": float(end), "n": touches, "st": status}
    return best


def _patterns(h, l, c, ph, pl, tol, look):
    last = len(c) - 1
    start = last - look
    ph = [i for i in ph if i >= start]
    pl = [i for i in pl if i >= start]
    out = []
    close = c[-1]
    # ダブルボトム
    if len(pl) >= 2:
        a, b = pl[-2], pl[-1]
        if b - a >= 6 and abs(l[a] - l[b]) / min(l[a], l[b]) <= tol * 2 and last - b <= look // 2:
            neck = h[a : b + 1].max()
            if neck >= min(l[a], l[b]) * (1 + tol * 3):
                if close > neck:
                    out.append("ダブルボトム(ブレイク)")
                elif close > l[b]:
                    out.append("ダブルボトム(形成中)")
    # ダブルトップ
    if len(ph) >= 2:
        a, b = ph[-2], ph[-1]
        if b - a >= 6 and abs(h[a] - h[b]) / max(h[a], h[b]) <= tol * 2 and last - b <= look // 2:
            neck = l[a : b + 1].min()
            if neck <= max(h[a], h[b]) * (1 - tol * 3):
                if close < neck:
                    out.append("ダブルトップ(割れ)")
                elif close < h[b]:
                    out.append("ダブルトップ(形成中)")
    if len(ph) >= 2 and len(pl) >= 2:
        hh = h[ph[-1]] > h[ph[-2]]
        hl = l[pl[-1]] > l[pl[-2]]
        lh = h[ph[-1]] < h[ph[-2]]
        ll = l[pl[-1]] < l[pl[-2]]
        if hh and hl:
            out.append("高値・安値切り上げ")
        elif lh and ll:
            out.append("高値・安値切り下げ")
        elif lh and hl:
            out.append("三角保ち合い(ブレイク)" if close > h[ph[-1]] else "三角保ち合い")
        # ボックス: 直近の山と谷がそれぞれ揃っている
        if len(ph) >= 3 and len(pl) >= 3:
            hs = h[ph[-3:]]
            ls = l[pl[-3:]]
            if hs.max() / hs.min() - 1 <= tol * 2 and ls.max() / ls.min() - 1 <= tol * 2:
                out.append("ボックス")
    return out


def analyze(df: pd.DataFrame, tf: str) -> dict:
    p = PARAMS[tf]
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    c = df["Close"].to_numpy(float)
    dates = [int(x.strftime("%Y%m%d")) for x in df.index]
    n = len(c)
    res = {"sr": [], "tl": [], "fib": None, "pat": []}
    if n < p["k"] * 2 + 5:
        return res
    last = n - 1
    close = c[-1]
    ph, pl = pivots(h, l, p["k"])
    start = max(0, n - p["look"])

    # サポート・レジスタンス(現在値の上下それぞれ近い順に3本)
    prices = [h[i] for i in ph if i >= start] + [l[i] for i in pl if i >= start]
    lv = _sr_levels(prices, p["tol"])
    sup = sorted([x for x in lv if x["p"] < close], key=lambda x: close - x["p"])[:3]
    res_ = sorted([x for x in lv if x["p"] >= close], key=lambda x: x["p"] - close)[:3]
    res["sr"] = [{"p": round(x["p"], 1), "n": x["n"], "t": "s"} for x in sup] + [
        {"p": round(x["p"], 1), "n": x["n"], "t": "r"} for x in res_
    ]

    # トレンドライン
    up = _trendline([i for i in pl if i >= start], l, h, c, True, p["tol"], last)
    dn = _trendline([i for i in ph if i >= start], l, h, c, False, p["tol"], last)
    for t, kind in ((up, "up"), (dn, "down")):
        if t:
            res["tl"].append({
                "k": kind, "d1": dates[t["a"]], "p1": round(t["ya"], 1), "d2": dates[last],
                "p2": round(t["end"], 1), "n": t["n"], "st": t["st"],
            })

    # フィボナッチ(直近の高値と安値)
    fs = max(0, n - p["fib"])
    hi_i = fs + int(h[fs:].argmax())
    lo_i = fs + int(l[fs:].argmin())
    hi, lo = h[hi_i], l[lo_i]
    if hi > lo:
        upmove = lo_i < hi_i  # 安値→高値の順なら上昇の押し目を測る
        levels = [round(hi - (hi - lo) * r, 1) if upmove else round(lo + (hi - lo) * r, 1) for r in FIB_RATIOS]
        res["fib"] = {
            "dir": "up" if upmove else "down",
            "hi": round(hi, 1), "lo": round(lo, 1),
            "dhi": dates[hi_i], "dlo": dates[lo_i],
            "lv": levels,
        }

    res["pat"] = _patterns(h, l, c, ph, pl, p["tol"], p["pat"])
    return res


def screen_flags(daily_analysis: dict, close: float) -> dict:
    """スクリーニング用の判定(日足ベース)"""
    a = daily_analysis
    sup = [x for x in a["sr"] if x["t"] == "s"]
    near_sup = None
    if sup:
        near_sup = round((close / sup[0]["p"] - 1) * 100, 2)
    tl_up = None
    for t in a["tl"]:
        if t["k"] == "up" and t["p2"] > 0 and t.get("st") != "break":
            tl_up = round((close / t["p2"] - 1) * 100, 2)
    fib = False
    if a["fib"] and a["fib"]["dir"] == "up":
        lv = a["fib"]["lv"]
        fib = lv[4] <= close <= lv[2]  # 38.2%〜61.8%押し
    return {
        "sup": near_sup,          # 直近サポートから何%上か
        "tlu": tl_up,             # 上昇トレンドラインから何%上か
        "fib": bool(fib),         # フィボナッチ38.2〜61.8%押しの範囲内
        "pat": "|".join(a["pat"]),
    }
