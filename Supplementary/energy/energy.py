#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute CPU package energy from an HWiNFO CSV log using a run window file.

What I do here:
- I read run_window.txt to get duration = (t1 - t0) and N (number of fits).
- I read the HWiNFO CSV (encoding-robust), detect Date/Time/Power columns.
- I parse HWiNFO "Time" as elapsed time (mm:ss.s / h:mm:ss.s / ss.s).
- I DO NOT align epoch time with HWiNFO wall clock; I only use duration.
- I pick a window (first duration vs last duration) and integrate power over time.

Outputs:
- Total energy (J)
- Energy per fit (J/fit)
- Optionally save the selected window samples
"""

import argparse
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


ENCODING_CANDIDATES = ["utf-8-sig", "utf-8", "gb18030", "gbk", "cp1252", "latin1"]


# -------------------------
# I/O helpers
# -------------------------
def read_run_window(path: str) -> Tuple[float, float, int]:
    """Read t0, t1 (epoch seconds) and N from run_window.txt."""
    with open(path, "r", encoding="utf-8") as f:
        t0 = float(f.readline().strip())
        t1 = float(f.readline().strip())
        n = int(f.readline().strip())

    if t1 <= t0:
        raise ValueError(f"Invalid run window: t1 <= t0 (t0={t0}, t1={t1}).")
    if n <= 0:
        raise ValueError(f"Invalid N in run window: N={n}.")
    return t0, t1, n


def try_read_csv_header(csv_path: str) -> Tuple[List[str], str]:
    """Read only the header row and return (columns, encoding_used)."""
    last_err = None
    for enc in ENCODING_CANDIDATES:
        try:
            hdr = pd.read_csv(csv_path, encoding=enc, nrows=0)
            return list(hdr.columns.astype(str)), enc
        except Exception as e:
            last_err = e
    raise RuntimeError(
        f"Cannot read CSV header. Tried encodings {ENCODING_CANDIDATES}. Last error: {last_err}"
    )


def robust_read_csv(
    csv_path: str,
    usecols: List[str],
    encoding_hint: Optional[str] = None,
) -> Tuple[pd.DataFrame, str]:
    """
    Read CSV with encoding fallback.
    - I try the fast C engine first.
    - If that fails, I fall back to the python engine.
    """
    enc_list = []
    if encoding_hint:
        enc_list.append(encoding_hint)
    enc_list += [e for e in ENCODING_CANDIDATES if e != encoding_hint]

    last_err = None
    for enc in enc_list:
        # keep dtype=str so I control conversions explicitly later
        try:
            df = pd.read_csv(csv_path, encoding=enc, usecols=usecols, dtype=str, engine="c")
            return df, enc
        except Exception as e_c:
            try:
                df = pd.read_csv(csv_path, encoding=enc, usecols=usecols, dtype=str, engine="python")
                return df, enc
            except Exception as e_py:
                last_err = (e_c, e_py)

    raise RuntimeError(f"CSV read failed. Tried encodings {enc_list}. Last error: {last_err}")


# -------------------------
# Column detection
# -------------------------
def find_date_time_cols(cols: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """Prefer exact 'Date'/'Time', otherwise do a loose match."""
    date_col = None
    time_col = None

    if "Date" in cols:
        date_col = "Date"
    else:
        for c in cols:
            lc = c.lower()
            if lc == "date" or "date" in lc or "日期" in c:
                date_col = c
                break

    if "Time" in cols:
        time_col = "Time"
    else:
        for c in cols:
            lc = c.lower()
            if lc == "time" or "time" in lc or "时间" in c:
                time_col = c
                break

    return date_col, time_col


def _score_power_col(name: str) -> Tuple[int, int, int, int]:
    """
    Lower is better (tuple sorting).
    I strongly prefer CPU package power, and I mildly prefer explicit units.
    """
    ln = name.lower()

    good = 0
    if "cpu package power" in ln:
        good += 10
    if "cpu 封装功率" in name:
        good += 10

    has_unit = 1 if ("[w]" in ln or "(w)" in ln) else 0

    bad = 0
    for kw in ["vr vcc", "pout", "ia core power", "system agent", "rest-of-chip", "剩余芯片功率", "ia 核心功率"]:
        if kw in ln:
            bad += 1

    return (-good, -has_unit, bad, len(name))


def find_power_col(cols: List[str], override: Optional[str] = None) -> str:
    """Find the best-matching CPU package power column (or use user override)."""
    if override:
        if override not in cols:
            raise ValueError(f"--power_col '{override}' not found in CSV columns.")
        return override

    candidates = []
    for c in cols:
        lc = c.lower()
        if "cpu package power" in lc or "cpu 封装功率" in c:
            candidates.append(c)

    if not candidates:
        for c in cols:
            lc = c.lower()
            if ("package" in lc and "power" in lc) or ("封装" in c and "功率" in c):
                candidates.append(c)

    if not candidates:
        raise ValueError("No CPU package power column found (e.g., 'CPU Package Power' / 'CPU 封装功率 [W]').")

    return sorted(candidates, key=_score_power_col)[0]


# -------------------------
# Parsing & cleaning
# -------------------------
def clean_repeated_headers(df: pd.DataFrame, date_col: Optional[str], time_col: str) -> pd.DataFrame:
    """
    HWiNFO logs sometimes repeat 'Date,Time' lines inside the CSV; I drop them.
    """
    out = df.copy()
    if date_col and date_col in out.columns:
        dc = out[date_col].astype(str).str.strip()
        out = out[(dc != "Date") & (dc != "日期")]

    tc = out[time_col].astype(str).str.strip()
    out = out[(tc != "Time") & (tc != "时间")]
    return out


def parse_hwinfo_time_to_seconds(time_series: pd.Series) -> np.ndarray:
    """
    Parse HWiNFO Time as elapsed seconds.

    Accepts:
      - "26:49.9"    -> mm:ss.s  (I normalize to 00:mm:ss.s)
      - "1:02:03.4"  -> h:mm:ss.s
      - "13:45:01"   -> h:mm:ss
      - "59.2"       -> ss.s     (I normalize to 00:00:ss.s)
    """
    t = time_series.astype(str).str.strip()
    colon_cnt = t.str.count(":")

    t_fixed = t.copy()
    t_fixed[colon_cnt == 1] = "00:" + t_fixed[colon_cnt == 1]
    t_fixed[colon_cnt == 0] = "00:00:" + t_fixed[colon_cnt == 0]

    td = pd.to_timedelta(t_fixed, errors="coerce")
    if td.isna().all():
        sample = time_series.dropna().astype(str).head(10).tolist()
        raise ValueError(f"Time column cannot be parsed as timedelta. Sample (first 10): {sample}")

    return td.dt.total_seconds().to_numpy()


# -------------------------
# Window selection & integration
# -------------------------
def pick_window_by_duration(
    t_sec: np.ndarray,
    p_w: np.ndarray,
    duration: float,
    force: str = "AUTO",
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    I only use duration, then pick:
      A) first [tmin, tmin+duration]
      B) last  [tmax-duration, tmax]
    In AUTO mode, I pick the one with higher mean power (and use points count as tie-breaker).
    """
    tmin = float(np.nanmin(t_sec))
    tmax = float(np.nanmax(t_sec))

    mask_a = (t_sec >= tmin) & (t_sec <= tmin + duration)
    mask_b = (t_sec >= tmax - duration) & (t_sec <= tmax)

    def _stats(mask: np.ndarray) -> Tuple[int, float]:
        x = t_sec[mask]
        y = p_w[mask]
        ok = np.isfinite(x) & np.isfinite(y)
        x = x[ok]
        y = y[ok]
        if len(x) < 2:
            return 0, -np.inf
        return int(len(x)), float(np.nanmean(y))

    if force == "A":
        return t_sec[mask_a], p_w[mask_a], "A(forced)"
    if force == "B":
        return t_sec[mask_b], p_w[mask_b], "B(forced)"

    n_a, m_a = _stats(mask_a)
    n_b, m_b = _stats(mask_b)

    if n_a < 2 and n_b < 2:
        raise ValueError(
            f"Not enough samples in either window. duration={duration:.3f}s, "
            f"log range={tmin:.3f}~{tmax:.3f}s, pointsA={n_a}, pointsB={n_b}."
        )
    if n_a >= 2 and n_b < 2:
        return t_sec[mask_a], p_w[mask_a], "A(first duration; only valid)"
    if n_b >= 2 and n_a < 2:
        return t_sec[mask_b], p_w[mask_b], "B(last duration; only valid)"

    # both valid: prefer higher mean power; if within 5%, prefer more points
    if m_a > m_b * 1.05:
        return t_sec[mask_a], p_w[mask_a], "A(first duration; higher mean power)"
    if m_b > m_a * 1.05:
        return t_sec[mask_b], p_w[mask_b], "B(last duration; higher mean power)"

    if n_a >= n_b:
        return t_sec[mask_a], p_w[mask_a], "A(first duration; more/equal points)"
    return t_sec[mask_b], p_w[mask_b], "B(last duration; more points)"


def integrate_energy_joules(t_sec: np.ndarray, p_w: np.ndarray) -> Tuple[float, int]:
    """Trapezoidal integration: E = ∫ P(t) dt (W*s = J)."""
    ok = np.isfinite(t_sec) & np.isfinite(p_w)
    x = t_sec[ok].astype(float)
    y = p_w[ok].astype(float)

    if x.size < 2:
        raise ValueError("Not enough valid points to integrate.")

    # sort by time, then drop duplicate/non-increasing timestamps
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    keep = np.ones_like(x, dtype=bool)
    keep[1:] = x[1:] > x[:-1]
    x = x[keep]
    y = y[keep]

    if x.size < 2:
        raise ValueError("After removing duplicate timestamps, points are still insufficient.")

    e_j = float(np.trapezoid(y, x))
    return e_j, int(x.size)


# -------------------------
# Main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="hwinfo_log.csv", help="HWiNFO CSV log path")
    ap.add_argument("--window", default="run_window.txt", help="run_window.txt path")
    ap.add_argument("--power_col", default="", help="Optional: override power column name")
    ap.add_argument("--force_window", choices=["A", "B", "AUTO"], default="AUTO",
                    help="A=first duration, B=last duration, AUTO=heuristic pick")
    ap.add_argument("--save_window_csv", default="", help="Optional: save selected window samples to CSV")
    args = ap.parse_args()

    t0, t1, n_fits = read_run_window(args.window)
    duration = t1 - t0

    cols, enc_hint = try_read_csv_header(args.log)
    date_col, time_col = find_date_time_cols(cols)
    if time_col is None:
        raise ValueError("Cannot find Time/时间 column in the HWiNFO CSV.")

    power_col = find_power_col(cols, override=(args.power_col or None))

    print("[INFO] Detected columns:")
    print(f"       Date col : {date_col}")
    print(f"       Time col : {time_col}")
    print(f"       Power col: {power_col}")
    print(f"[INFO] Using duration from run_window: {duration:.6f} s, N(fits)={n_fits}")

    usecols = [time_col, power_col] if not date_col else [date_col, time_col, power_col]
    df, enc_used = robust_read_csv(args.log, usecols=usecols, encoding_hint=enc_hint)
    print(f"[INFO] CSV read ok. encoding={enc_used}, rows={len(df)}")

    df = clean_repeated_headers(df, date_col=date_col, time_col=time_col)

    t_sec = parse_hwinfo_time_to_seconds(df[time_col])
    p_w = pd.to_numeric(df[power_col], errors="coerce").to_numpy(dtype=float)

    tw, pw, tag = pick_window_by_duration(t_sec, p_w, duration, force=args.force_window)
    e_j, npts = integrate_energy_joules(tw, pw)

    print(f"[INFO] Selected window: {tag}")
    print(f"[INFO] Window points used for integration: {npts}")
    print(f"Total energy (J): {e_j:.6f}")
    print(f"Energy per fit (J/fit): {e_j / n_fits:.9f}")

    if args.save_window_csv:
        out = pd.DataFrame({"t_sec": tw, "power_W": pw})
        out = out[np.isfinite(out["t_sec"]) & np.isfinite(out["power_W"])].sort_values("t_sec")
        out.to_csv(args.save_window_csv, index=False, encoding="utf-8-sig")
        print(f"[INFO] Saved selected window to: {args.save_window_csv}")


if __name__ == "__main__":
    main()


