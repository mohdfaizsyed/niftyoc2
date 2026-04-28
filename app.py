from flask import Flask, render_template, jsonify, send_file
import sqlite3
import csv
import requests
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta

app = Flask(__name__)

# =====================================================
# DATABASE
# =====================================================
DB_FILE = "nima_history.db"


def init_db():
    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS history(
        trade_date TEXT,
        trade_time TEXT,
        spot REAL,
        eor REAL,
        eos REAL,
        eor_plus_1 REAL,
        eor_minus_1 REAL,
        eos_plus_1 REAL,
        eos_minus_1 REAL,
        market_pcr REAL,
        put_writers REAL,
        call_writers REAL,
        PRIMARY KEY(trade_date, trade_time)
    )
    """)

    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_trade_date_time
    ON history(trade_date, trade_time)
    """)

    con.commit()
    con.close()


init_db()

# =====================================================
# SAVE HISTORY
# =====================================================
def save_history(payload):
    now = datetime.now()

    # skip weekend
    if now.weekday() >= 5:
        return

    hhmm = now.strftime("%H:%M")

    # market hours
    if hhmm < "09:15" or hhmm > "15:30":
        return

    trade_date = now.strftime("%Y-%m-%d")
    trade_time = now.strftime("%H:%M")

    spot = float(payload.get("spot", 0))
    eor = float(payload.get("eor", 0))
    eos = float(payload.get("eos", 0))
    pcr = float(payload.get("market_pcr", 0))
    putw = float(payload.get("put_writers_percent", 0))
    callw = float(payload.get("call_writers_percent", 0))

    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()

    cur.execute("""
        SELECT spot,eor,eos,market_pcr,put_writers,call_writers
        FROM history
        WHERE trade_date=? AND trade_time=?
    """, (trade_date, trade_time))

    old = cur.fetchone()

    # prevent duplicate unchanged row
    if old:
        if (
            float(old[0]) == spot and
            float(old[1]) == eor and
            float(old[2]) == eos and
            float(old[3]) == pcr and
            float(old[4]) == putw and
            float(old[5]) == callw
        ):
            con.close()
            return

    cur.execute("""
        INSERT OR REPLACE INTO history
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        trade_date,
        trade_time,
        spot,
        eor,
        eos,
        float(payload.get("eor_plus_1", 0)),
        float(payload.get("eor_minus_1", 0)),
        float(payload.get("eos_plus_1", 0)),
        float(payload.get("eos_minus_1", 0)),
        pcr,
        putw,
        callw
    ))

    cur.execute("""
    CREATE TABLE IF NOT EXISTS signal_log(
        trade_date TEXT,
        trade_time TEXT,
        signal TEXT,
        PRIMARY KEY(trade_date, trade_time)
    )
    """)
   

    # keep last 7 days
    cur.execute("""
        DELETE FROM history
        WHERE datetime(trade_date || ' ' || trade_time)
        < datetime('now','-7 days')
    """)

    con.commit()
    con.close()

def save_signal(signal):

    # log only actionable signals
    if signal not in ["CALL", "PUT"]:
        return

    now = datetime.now()

    if now.weekday() >= 5:
        return

    hhmm = now.strftime("%H:%M")

    if hhmm < "09:15" or hhmm > "15:30":
        return

    trade_date = now.strftime("%Y-%m-%d")
    trade_time = now.strftime("%H:%M")

    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()

    cur.execute("""
        INSERT OR REPLACE INTO signal_log
        VALUES(?,?,?)
    """, (trade_date, trade_time, signal))

    # keep 7 days only
    cur.execute("""
        DELETE FROM signal_log
        WHERE datetime(trade_date || ' ' || trade_time)
        < datetime('now','-7 days')
    """)

    con.commit()
    con.close()


# =====================================================
# TREND ENGINE
# =====================================================
def get_nearest_row(cur, trade_date, target_time):
    cur.execute("""
        SELECT *
        FROM history
        WHERE trade_date=? AND trade_time<=?
        ORDER BY trade_time DESC
        LIMIT 1
    """, (trade_date, target_time))

    return cur.fetchone()


def calc_pct_trend(current, old, threshold):
    try:
        if old in (None, 0):
            return "STABLE"

        pct = ((current - old) / old) * 100

        if pct > threshold:
            return "RISING"

        if pct < -threshold:
            return "FALLING"

        return "STABLE"

    except:
        return "STABLE"


def calc_shift(current, old, points=60):
    try:
        if old is None:
            return "STABLE"

        diff = current - old

        if diff > points:
            return "UP SHIFT"

        if diff < -points:
            return "DOWN SHIFT"

        return "STABLE"

    except:
        return "STABLE"


def get_db_trends():
    con = sqlite3.connect(DB_FILE)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    # -----------------------------------
    # latest available trading date
    # -----------------------------------
    cur.execute("""
        SELECT MAX(trade_date) AS d
        FROM history
    """)
    row = cur.fetchone()

    if not row or not row["d"]:
        con.close()
        return {}

    trade_date = row["d"]

    # -----------------------------------
    # first row of that day = baseline
    # (whenever system started recording)
    # -----------------------------------
    cur.execute("""
        SELECT *
        FROM history
        WHERE trade_date=?
        ORDER BY trade_time ASC
        LIMIT 1
    """, (trade_date,))
    first = cur.fetchone()

    # -----------------------------------
    # latest row of same day
    # -----------------------------------
    cur.execute("""
        SELECT *
        FROM history
        WHERE trade_date=?
        ORDER BY trade_time DESC
        LIMIT 1
    """, (trade_date,))
    latest = cur.fetchone()

    if not first or not latest:
        con.close()
        return {}

    # -----------------------------------
    # helper functions
    # -----------------------------------
    def pct_change(now, base):
        try:
            now = float(now)
            base = float(base)

            if base == 0:
                return 0.0

            return round(((now - base) / base) * 100, 1)

        except:
            return 0.0

    def point_change(now, base):
        try:
            return int(round(float(now) - float(base), 0))
        except:
            return 0

    # -----------------------------------
    # quantified deltas
    # -----------------------------------
    # -----------------------------------
    # first non-zero baselines
    # -----------------------------------

    # PCR baseline
    cur.execute("""
        SELECT market_pcr
        FROM history
        WHERE trade_date=? AND market_pcr > 0
        ORDER BY trade_time ASC
        LIMIT 1
    """, (trade_date,))
    row_pcr = cur.fetchone()

    # Put Writers baseline
    cur.execute("""
        SELECT put_writers
        FROM history
        WHERE trade_date=? AND put_writers > 0
        ORDER BY trade_time ASC
        LIMIT 1
    """, (trade_date,))
    row_pw = cur.fetchone()

    # Call Writers baseline
    cur.execute("""
        SELECT call_writers
        FROM history
        WHERE trade_date=? AND call_writers > 0
        ORDER BY trade_time ASC
        LIMIT 1
    """, (trade_date,))
    row_cw = cur.fetchone()

    pcr_base = row_pcr["market_pcr"] if row_pcr else first["market_pcr"]
    put_base = row_pw["put_writers"] if row_pw else first["put_writers"]
    call_base = row_cw["call_writers"] if row_cw else first["call_writers"]

    market_pcr_delta = pct_change(
        latest["market_pcr"],
        pcr_base
    )

    put_writers_delta = pct_change(
        latest["put_writers"],
        put_base
    )

    call_writers_delta = pct_change(
        latest["call_writers"],
        call_base
    )

    spot_delta = point_change(
        latest["spot"],
        first["spot"]
    )

    eor_delta = point_change(
        latest["eor"],
        first["eor"]
    )

    eos_delta = point_change(
        latest["eos"],
        first["eos"]
    )

    # -----------------------------------
    # flow summary using quantified signs
    # -----------------------------------
    flow = "Mixed Flow"
    flow_class = "strong"

    if (
        market_pcr_delta > 0 and
        put_writers_delta > 0 and
        call_writers_delta < 0
    ):
        flow = "strongest bullish"
        flow_class = "up"

    elif (
        market_pcr_delta < 0 and
        put_writers_delta < 0 and
        call_writers_delta > 0
    ):
        flow = "strongest bearish"
        flow_class = "down"

    elif (
        market_pcr_delta > 0 and
        call_writers_delta < 0
    ):
        flow = "bullish bounce-Short Covering"
        flow_class = "up"

    elif (
        market_pcr_delta < 0 and
        call_writers_delta > 0
    ):
        flow = "Bearish Bias"
        flow_class = "down"

    elif (
        abs(market_pcr_delta) <= 1 and
        abs(put_writers_delta) <= 1 and
        abs(call_writers_delta) <= 1
    ):
        flow = "Neutral / Rangebound"
        flow_class = "strong"

    
       # -----------------------------------
    # Previous available trading date
    # -----------------------------------
    cur.execute("""
        SELECT trade_date
        FROM history
        WHERE trade_date < ?
        GROUP BY trade_date
        ORDER BY trade_date DESC
        LIMIT 1
    """, (trade_date,))

    row = cur.fetchone()

    prev_date = row["trade_date"] if row else None

    y_high = None
    y_low = None
    y_close = None

    if prev_date:

        cur.execute("""
            SELECT MAX(spot), MIN(spot)
            FROM history
            WHERE trade_date=?
            AND spot > 1000
        """, (prev_date,))

        row2 = cur.fetchone()

        if row2:
            y_high = row2[0]
            y_low = row2[1]

        cur.execute("""
            SELECT spot
            FROM history
            WHERE trade_date=?
            AND spot > 1000
            ORDER BY trade_time DESC
            LIMIT 1
        """, (prev_date,))

        row3 = cur.fetchone()

        if row3:
            y_close = row3[0]

    con.close()

    return {
        "market_pcr_delta": market_pcr_delta,
        "put_writers_delta": put_writers_delta,
        "call_writers_delta": call_writers_delta,
        "spot_delta": spot_delta,
        "eor_delta": eor_delta,
        "eos_delta": eos_delta,
        "flow_summary": flow,
        "flow_class": flow_class,
        "y_high": y_high if y_high not in (None,0) else None,
        "y_low": y_low if y_low not in (None,0) else None,
        "y_close": y_close if y_close not in (None,0) else None
    }

# -------------------------------------------------
# NSE SESSION
# -------------------------------------------------
session = requests.Session()

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/option-chain"
}

# -------------------------------------------------
# SETTINGS
# -------------------------------------------------
RISK_FREE_RATE = 0.10
DIVIDEND_YIELD = 0.01
DAYS_IN_YEAR = 365


# -------------------------------------------------
# FETCH JSON
# -------------------------------------------------
def get_json(url):
    try:
        session.get("https://www.nseindia.com", headers=HEADERS, timeout=5)
        r = session.get(url, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return {}


# -------------------------------------------------
# NEXT TUESDAY EXPIRY
# -------------------------------------------------
def get_current_expiry():
    today = datetime.now()
    days_ahead = 1 - today.weekday()  # Tuesday = 1

    if days_ahead < 0:
        days_ahead += 7

    expiry = today + timedelta(days=days_ahead)
    return expiry.strftime("%d-%b-%Y")


# -------------------------------------------------
# CLEAN JSON
# -------------------------------------------------
def clean(v):
    if pd.isna(v):
        return 0
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


# -------------------------------------------------
# DELTA
# -------------------------------------------------
def calculate_delta(S, K, r, q, sigma, T, option_type):
    try:
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0.0

        d1 = (
            np.log(S / K) +
            (r - q + (sigma ** 2) / 2) * T
        ) / (sigma * np.sqrt(T))

        if option_type == "CE":
            return norm.cdf(d1) / 1

        if option_type == "PE":
            return (norm.cdf(d1) - 1) / 1

        return 0.0
    except:
        return 0.0


# -------------------------------------------------
# EXT LOOKUP
# -------------------------------------------------
def get_ext(df, strike, col):
    try:
        if strike in df["Strike"].values:
            val = df.loc[df["Strike"] == strike, col].iloc[0]
            return int(round(float(val), 0))
        return None
    except:
        return None


# -------------------------------------------------
# RESISTANCE TENDENCY
# -------------------------------------------------
def resistance_tendency(rtype, resistance, oi_strike, oi_pct, vol_strike, vol_pct):

    if rtype == "VOL":
        if vol_pct > 75:
            return "UP" if vol_strike > resistance else "DOWN"

    elif rtype == "OI":
        if oi_pct > 75:
            return "UP" if oi_strike > resistance else "DOWN"

    elif rtype == "Both OI VOL":

        if oi_pct > 75 and vol_pct > 75:
            if oi_strike > resistance and vol_strike > resistance:
                return "UP"

            if oi_strike < resistance or vol_strike < resistance:
                return "DOWN"

    return "STRONG"


# -------------------------------------------------
# SUPPORT TENDENCY
# -------------------------------------------------
def support_tendency(stype, support, oi_strike, oi_pct, vol_strike, vol_pct):

    if stype == "VOL":
        if vol_pct > 75:
            return "UP" if vol_strike > support else "DOWN"

    elif stype == "OI":
        if oi_pct > 75:
            return "UP" if oi_strike > support else "DOWN"

    elif stype == "Both OI VOL":

        if oi_pct > 75 and vol_pct > 75:
            if oi_strike < support and vol_strike < support:
                return "DOWN"

            if oi_strike > support or vol_strike > support:
                return "UP"

    return "STRONG"

def calculate_max_pain(df2):
    try:
        temp = df2.copy()

        temp["Strike Price"] = temp["Strike"]
        temp = temp.sort_values(by="Strike Price")

        temp["call_value"] = (
            temp["CE OI"] *
            (temp["Strike Price"].max() - temp["Strike Price"])
        )

        temp["put_value"] = (
            temp["PE OI"] *
            (temp["Strike Price"] - temp["Strike Price"].min())
        )

        temp["sum"] = temp["call_value"] + temp["put_value"]

        max_pain_strike = temp.loc[
            temp["sum"].idxmax(),
            "Strike Price"
        ]

        return int(max_pain_strike)

    except:
        return 0

def generate_trade_signal(
    support_tend,
    resistance_tend,
    spot,

    market_pcr_delta,
    put_writers_delta,
    call_writers_delta,

    support,
    resistance,

    eos,
    eor,

    support_tendency_strike,
    resistance_tendency_strike
):

    # ==========================================
    # CALL PLAN 1  (Your Requested Setup)
    # ==========================================
    if (
        resistance_tendency_strike >= resistance and
        support_tendency_strike >= support and
        market_pcr_delta >= 5 and
        put_writers_delta >= 5 and
        resistance_tend != "DOWN" and
        support_tend != "DOWN"
        #spot < (eos + 15)
    ):

        rng = int(abs(eor - spot))

        return (
            "CALL",
            f"Below {int(eos + 15)}",
            int(eor),
            rng
        )

    
    # ==========================================
    # PUT PLAN 1
    # ==========================================
    if (
        resistance_tendency_strike <= resistance and
        support_tendency_strike <= support and
        market_pcr_delta <= -5 and
        call_writers_delta >= 5 and
        resistance_tend != "UP" and
        support_tend != "UP"
        #spot > (eor - 15)
    ):

        rng = int(abs(spot - eos))

        return (
            "PUT",
            f"Above {int(eor - 15)}",
            int(eos),
            rng
        )

    
    # ==========================================
    # DEFAULT
    # ==========================================
    return ("WAIT", "", "", "")

# -------------------------------------------------
# MAIN ENGINE
# -------------------------------------------------
def fetch_option_chain():

    expiry = get_current_expiry()

    url = f"https://www.nseindia.com/api/option-chain-v3?type=Indices&symbol=NIFTY&expiry={expiry}"

    data = get_json(url)

    try:
        records = data["records"]["data"]
        spot = data["records"]["underlyingValue"]
    except:
        return {}

    current_date = datetime.now()
    expiry_date = datetime.strptime(expiry, "%d-%b-%Y")
    T = max((expiry_date - current_date).days / 365, 0.01)

    rows = []
    strikes = []

    # -------------------------------------------------
    # BUILD DATAFRAME
    # -------------------------------------------------
    for item in records:

        strike = item.get("strikePrice")
        ce = item.get("CE", {})
        pe = item.get("PE", {})

        ce_iv = ce.get("impliedVolatility", 0)
        pe_iv = pe.get("impliedVolatility", 0)

        ce_ltp = ce.get("lastPrice", 0)
        pe_ltp = pe.get("lastPrice", 0)

        ce_sigma = ce_iv / 100 if ce_iv > 0 else 0.01
        pe_sigma = pe_iv / 100 if pe_iv > 0 else 0.01

        ce_delta = round(
            calculate_delta(
                spot, strike, RISK_FREE_RATE,
                DIVIDEND_YIELD, ce_sigma, T, "CE"
            ), 4
        )

        pe_delta = round(
            calculate_delta(
                spot, strike, RISK_FREE_RATE,
                DIVIDEND_YIELD, pe_sigma, T, "PE"
            ), 4
        )

        ce_ext = round((ce_delta * ce_ltp * 0.2) + strike, 0)
        pe_ext = round((pe_delta * pe_ltp * 0.2) + strike, 0)

        strikes.append(strike)

        rows.append({
            "Strike": strike,

            "CE OI": ce.get("openInterest", 0),
            "CE VOL": ce.get("totalTradedVolume", 0),
            "CE CHG OI": ce.get("changeinOpenInterest", 0),
            "CE Delta": ce_delta,
            "CE EXT": ce_ext,
            "CE IV": ce_iv,
            "CE LTP": ce_ltp,

            "PE OI": pe.get("openInterest", 0),
            "PE VOL": pe.get("totalTradedVolume", 0),
            "PE CHG OI": pe.get("changeinOpenInterest", 0),
            "PE Delta": pe_delta,
            "PE EXT": pe_ext,
            "PE IV": pe_iv,
            "PE LTP": pe_ltp
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("Strike").reset_index(drop=True)

    # -------------------------------------------------
    # ATM RANGE
    # -------------------------------------------------
    atm = min(strikes, key=lambda x: abs(x - spot))
    atm_index = df[df["Strike"] == atm].index[0]

    start = max(0, atm_index - 9)
    end = min(len(df), atm_index + 10)

    df2 = df.iloc[start:end].copy()
    max_pain = calculate_max_pain(df2)

    lower_limit = df2["Strike"].min()
    upper_limit = df2["Strike"].max()

    hstrike = min([s for s in strikes if s > spot], default=atm)
    lstrike = max([s for s in strikes if s < spot], default=atm)

    # -------------------------------------------------
    # FILTERED SUPPORT / RESISTANCE RANGE
    # -------------------------------------------------
    call_df = df[
        (df["Strike"] >= lstrike) &
        (df["Strike"] <= upper_limit)
    ]

    put_df = df[
        (df["Strike"] <= hstrike) &
        (df["Strike"] >= lower_limit)
    ]

    # -------------------------------------------------
    # RESISTANCE
    # -------------------------------------------------
    max_call_oi = call_df["CE OI"].max()
    max_call_vol = call_df["CE VOL"].max()

    max_call_oi_strike = call_df.loc[call_df["CE OI"].idxmax(), "Strike"]
    max_call_vol_strike = call_df.loc[call_df["CE VOL"].idxmax(), "Strike"]

    sec_call_oi = call_df["CE OI"].nlargest(2).iloc[-1]
    sec_call_vol = call_df["CE VOL"].nlargest(2).iloc[-1]

    sec_call_oi_strike = call_df.loc[call_df["CE OI"].nlargest(2).index[-1], "Strike"]
    sec_call_vol_strike = call_df.loc[call_df["CE VOL"].nlargest(2).index[-1], "Strike"]

    sec_call_oi_pct = round((sec_call_oi / max_call_oi) * 100, 2)
    sec_call_vol_pct = round((sec_call_vol / max_call_vol) * 100, 2)

    if max_call_oi_strike == max_call_vol_strike:
        resistance = max_call_oi_strike
        resistance_type = "Both OI VOL"
    elif max_call_oi_strike > max_call_vol_strike:
        resistance = max_call_vol_strike
        resistance_type = "VOL"
    else:
        resistance = max_call_oi_strike
        resistance_type = "OI"

    resistance_tend = resistance_tendency(
        resistance_type, resistance,
        sec_call_oi_strike, sec_call_oi_pct,
        sec_call_vol_strike, sec_call_vol_pct
    )

    # -------------------------------------------------
    # SUPPORT
    # -------------------------------------------------
    max_put_oi = put_df["PE OI"].max()
    max_put_vol = put_df["PE VOL"].max()

    max_put_oi_strike = put_df.loc[put_df["PE OI"].idxmax(), "Strike"]
    max_put_vol_strike = put_df.loc[put_df["PE VOL"].idxmax(), "Strike"]

    sec_put_oi = put_df["PE OI"].nlargest(2).iloc[-1]
    sec_put_vol = put_df["PE VOL"].nlargest(2).iloc[-1]

    sec_put_oi_strike = put_df.loc[put_df["PE OI"].nlargest(2).index[-1], "Strike"]
    sec_put_vol_strike = put_df.loc[put_df["PE VOL"].nlargest(2).index[-1], "Strike"]

    sec_put_oi_pct = round((sec_put_oi / max_put_oi) * 100, 2)
    sec_put_vol_pct = round((sec_put_vol / max_put_vol) * 100, 2)

    if max_put_oi_strike == max_put_vol_strike:
        support = max_put_oi_strike
        support_type = "Both OI VOL"
    elif max_put_oi_strike > max_put_vol_strike:
        support = max_put_oi_strike
        support_type = "OI"
    else:
        support = max_put_vol_strike
        support_type = "VOL"

    support_tend = support_tendency(
        support_type, support,
        sec_put_oi_strike, sec_put_oi_pct,
        sec_put_vol_strike, sec_put_vol_pct
    )

    # -------------------------------------------------
    # EOR / EOS
    # -------------------------------------------------
    eor = get_ext(df, resistance, "CE EXT")
    eor_plus_1 = get_ext(df, resistance + 50, "CE EXT")
    eor_minus_1 = get_ext(df, resistance - 50, "CE EXT")

    eos = get_ext(df, support, "PE EXT")
    eos_plus_1 = get_ext(df, support + 50, "PE EXT")
    eos_minus_1 = get_ext(df, support - 50, "PE EXT")

    # -------------------------------------------------
    # MARKET PCR
    # -------------------------------------------------
    total_pe_oi = df2["PE OI"].sum()
    total_ce_oi = df2["CE OI"].sum()

    market_pcr = round(total_pe_oi / total_ce_oi, 2) if total_ce_oi != 0 else 0

    # -------------------------------------------------
    # ATM PCR
    # -------------------------------------------------
    atm_row = df[df["Strike"] == atm]

    if not atm_row.empty and atm_row["CE OI"].iloc[0] != 0:
        atm_pcr = round(
            atm_row["PE OI"].iloc[0] / atm_row["CE OI"].iloc[0], 2
        )
    else:
        atm_pcr = 0

    # -------------------------------------------------
    # OI TRENDS (ATM ±3)
    # -------------------------------------------------
    seller_start = max(atm_index - 3, 0)
    seller_end = min(atm_index + 4, len(df))

    selected_rows = df.iloc[seller_start:seller_end].copy()

    # Positive = Writers
    pe_pos = selected_rows["PE CHG OI"].clip(lower=0).sum()
    ce_pos = selected_rows["CE CHG OI"].clip(lower=0).sum()

    pos_total = pe_pos + ce_pos

    if pos_total > 0:
        put_writers_percent = round((pe_pos / pos_total) * 100, 1)
        call_writers_percent = round((ce_pos / pos_total) * 100, 1)
    else:
        put_writers_percent = 0
        call_writers_percent = 0

    # Negative = Unwind
    pe_neg = abs(selected_rows["PE CHG OI"].clip(upper=0).sum())
    ce_neg = abs(selected_rows["CE CHG OI"].clip(upper=0).sum())

    neg_total = pe_neg + ce_neg

    if neg_total > 0:
        put_unwind_percent = round((pe_neg / neg_total) * 100, 1)
        call_unwind_percent = round((ce_neg / neg_total) * 100, 1)
    else:
        put_unwind_percent = 0
        call_unwind_percent = 0

    if put_writers_percent > call_writers_percent and call_unwind_percent > put_unwind_percent:
        net_bias = "Bullish"

    elif call_writers_percent > put_writers_percent and put_unwind_percent > call_unwind_percent:
        net_bias = "Bearish"

    else:
        net_bias = "Mixed"


# -----------------------------------
# MARKET INFLUENCE
# -----------------------------------
    bias_driver = "Balanced Flow"

    call_gap = call_unwind_percent - put_unwind_percent
    put_gap = put_unwind_percent - call_unwind_percent

    pw_gap = put_writers_percent - call_writers_percent
    cw_gap = call_writers_percent - put_writers_percent

    score_map = {
        "Resistance Weakening": call_gap,
        "Support Weakening": put_gap,
        "Support Strengthening": pw_gap,
        "Resistance Strengthening": cw_gap
    }

    best_key = max(score_map, key=score_map.get)

    if score_map[best_key] > 5:
        bias_driver = best_key


# -----------------------------------
# BIAS STRENGTH METER
# -----------------------------------
    bull_points = 0
    bear_points = 0

    # Writers
    bull_points += put_writers_percent
    bear_points += call_writers_percent

    # Unwind
    bull_points += call_unwind_percent * 0.6
    bear_points += put_unwind_percent * 0.6

    # PCR
    if market_pcr > 1:
        bull_points += min((market_pcr - 1) * 40, 20)
    elif market_pcr < 1:
        bear_points += min((1 - market_pcr) * 40, 20)

    # Tendencies
    if support_tend == "UP":
        bull_points += 10
    elif support_tend == "DOWN":
        bear_points += 10

    if resistance_tend == "DOWN":
        bull_points += 10
    elif resistance_tend == "UP":
        bear_points += 10

    total_points = bull_points + bear_points

    if total_points > 0:
        bull_strength = int((bull_points / total_points) * 100)
    else:
        bull_strength = 50

    if bull_strength >= 55:
        bias_strength = bull_strength
    elif bull_strength <= 45:
        bias_strength = 100 - bull_strength
    else:
        bias_strength = 50

    # ADD THIS BLOCK inside fetch_option_chain()
    # Place it AFTER OI Trends calculations
    # and BEFORE rows_json creation

    # -------------------------------------------------
# LIVE DIRECTION PREDICTOR
# -------------------------------------------------
    bull_score = 0
    bear_score = 0
    reasons = []

# Net Bias
    if net_bias == "Bullish":
        bull_score += 3
        reasons.append("Bullish OI Bias")
    elif net_bias == "Bearish":
        bear_score += 3
        reasons.append("Bearish OI Bias")

# Writers
    if put_writers_percent > 60:
        bull_score += 3
        reasons.append("Strong Put Writing")

    if call_writers_percent > 60:
        bear_score += 3
        reasons.append("Strong Call Writing")

# Unwind
    if call_unwind_percent > 60:
        bull_score += 3
        reasons.append("Call Unwinding")

    if put_unwind_percent > 60:
        bear_score += 3
        reasons.append("Put Unwinding")

# Tendencies
    if support_tend == "UP":
        bull_score += 3
        reasons.append("Support Rising")

    elif support_tend == "DOWN":
        bear_score += 3
        reasons.append("Support Weakening")

    if resistance_tend == "DOWN":
        bull_score += 2
        reasons.append("Resistance Weakening")

    elif resistance_tend == "UP":
        bear_score += 2
        reasons.append("Resistance Rising")

# PCR
    if market_pcr > 1:
        bull_score += 2
    elif market_pcr < 1:
        bear_score += 2

    if atm_pcr > 1:
        bull_score += 1
    elif atm_pcr < 1:
        bear_score += 1

# Spot vs ATM
    if spot > atm:
        bull_score += 1
    elif spot < atm:
        bear_score += 1

# Final Signal
    if bull_score >= 6 and bull_score > bear_score:
        predictor_signal = "BUY CE"

    elif bear_score >= 6 and bear_score > bull_score:
        predictor_signal = "BUY PE"

    elif abs(bull_score - bear_score) <= 1:
        predictor_signal = "RANGEBOUND"

    else:
        predictor_signal = "NO TRADE"

# Confidence
    max_score = max(bull_score, bear_score, 1)
    gap = abs(bull_score - bear_score)

    predictor_confidence = int((gap / max_score) * 100)

    predictor_reason = ", ".join(reasons[:4])
    # ---------------------------------------
    # TRADE SIGNAL ENGINE
    # ---------------------------------------
    support_tendency_strike = sec_put_oi_strike
    if sec_put_vol_pct > sec_put_oi_pct:
        support_tendency_strike = sec_put_vol_strike

    support_tendency_strike_ext = get_ext( df, support_tendency_strike, "PE EXT")
    support_tendency_strike_plus50_ext = get_ext( df, support_tendency_strike + 50, "PE EXT")

    
    resistance_tendency_strike = sec_call_oi_strike
    if sec_call_vol_pct > sec_call_oi_pct:
        resistance_tendency_strike = sec_call_vol_strike

    resistance_tendency_strike_ext = get_ext(df, resistance_tendency_strike, "CE EXT")

    resistance_tendency_strike_minus50_ext = get_ext( df, resistance_tendency_strike - 50, "CE EXT")
    

    tr = get_db_trends()

    market_pcr_delta = tr.get("market_pcr_delta", 0)
    put_writers_delta = tr.get("put_writers_delta", 0)
    call_writers_delta = tr.get("call_writers_delta", 0)

    trade_signal, trade_entry, trade_target, trade_range = generate_trade_signal(
        support_tend,
        resistance_tend,
        spot,

        market_pcr_delta,
        put_writers_delta,
        call_writers_delta,

        support,
        resistance,

        eos,
        eor,

        support_tendency_strike,
        resistance_tendency_strike
)

    save_signal(trade_signal)

    # -------------------------------------------------
    # TABLE JSON
    # -------------------------------------------------
    rows_json = []

    for _, row in df2.iterrows():
        obj = {}
        for c in df2.columns:
            obj[c] = clean(row[c])
        rows_json.append(obj)
    # -----------------------------------
    # SMART LADDER ANALYTICS
    # -----------------------------------

    levels = [
        ("UPPER RESISTANCE", eor_plus_1),
        ("RESISTANCE", eor),
        ("LOWER RESISTANCE", eor_minus_1),
        ("UPPER SUPPORT", eos_plus_1),
        ("SUPPORT", eos),
        ("LOWER SUPPORT", eos_minus_1)
    ]

    # nearest level to spot
    nearest_name = ""
    nearest_value = 0
    nearest_dist = 99999

    for nm, lv in levels:
        d = abs(float(spot) - float(lv))
        if d < nearest_dist:
            nearest_dist = d
            nearest_name = nm
            nearest_value = lv

    # -----------------------------------
    # CONFIDENCE SCORE (0-100)
    # -----------------------------------
    conf = 50

    # closer levels stronger
    if nearest_dist <= 20:
        conf += 20
    elif nearest_dist <= 40:
        conf += 12
    elif nearest_dist <= 70:
        conf += 6

    # tendency boosts
    if "RESISTANCE" in nearest_name:
        if resistance_tend == "STRONG":
            conf += 18
        elif resistance_tend == "UP":
            conf += 10
        elif resistance_tend == "DOWN":
            conf -= 10

    if "SUPPORT" in nearest_name:
        if support_tend == "STRONG":
            conf += 18
        elif support_tend == "UP":
            conf += 10
        elif support_tend == "DOWN":
            conf -= 10

    conf = max(1, min(99, int(conf)))

    # -----------------------------------
    # BREAK PROBABILITY
    # -----------------------------------
    brk = 50

    # bullish pressure
    if market_pcr > 1:
        brk += 8
    else:
        brk -= 8

    if put_writers_percent > call_writers_percent:
        brk += 6
    else:
        brk -= 6

    # level specific direction
    if "RESISTANCE" in nearest_name:
        if resistance_tend == "DOWN":
            brk += 18
        elif resistance_tend == "STRONG":
            brk -= 15

    if "SUPPORT" in nearest_name:
        if support_tend == "DOWN":
            brk += 18
        elif support_tend == "STRONG":
            brk -= 15

    # very near level => event likely soon
    if nearest_dist <= 25:
        brk += 10

    brk = max(1, min(99, int(brk)))

    # -----------------------------------
    # ACTION
    # -----------------------------------
    action = "WATCH"

    if "RESISTANCE" in nearest_name:
        if brk >= 65:
            action = "BREAK↑"
        elif conf >= 70:
            action = "HOLD"
        else:
            action = "WATCH"

    elif "SUPPORT" in nearest_name:
        if brk >= 65:
            action = "BREAK↓"
        elif conf >= 70:
            action = "HOLD"
        else:
            action = "WATCH"

    # -----------------------------------
    # OI DECISION ENGINE
    # -----------------------------------

    trade_mode = "NO EDGE"

    if net_bias == "Bullish" and bias_strength >= 60:
        trade_mode = "CALL BIAS"

    elif net_bias == "Bearish" and bias_strength >= 60:
        trade_mode = "PE BIAS"


    # Spot Confirmation
    spot_confirm = "⚠ Between Levels"

    if spot > eos and spot < eor:
        spot_confirm = "✓ Inside Range"

    if spot <= eos + 15:
        spot_confirm = "✓ Near Support"

    if spot >= eor - 15:
        spot_confirm = "✓ Near Resistance"

    if trade_mode == "CALL BIAS" and spot > eos:
        spot_confirm = "✓ Above Support"

    if trade_mode == "PE BIAS" and spot < eor:
        spot_confirm = "✓ Below Resistance"


    # Entry Confidence
    entry_conf = 50

    if trade_mode in ["CALL BIAS", "PE BIAS"]:
        entry_conf += 15

    entry_conf += int((bias_strength - 50) * 0.6)

    if "Above Support" in spot_confirm:
        entry_conf += 10

    if "Below Resistance" in spot_confirm:
        entry_conf += 10

    entry_conf = max(1, min(99, int(entry_conf)))


        # -------------------------------------------------
        # RETURN JSON
        # -------------------------------------------------
    return {
        "spot": clean(spot),
        "atm": clean(atm),
        "expiry": expiry,
        "updated": datetime.now().strftime("%H:%M:%S"),
        "market_pcr": market_pcr,
        "trade_signal": trade_signal,
        "trade_entry": trade_entry,
        "trade_target": trade_target,
        "trade_range": trade_range,
        "atm_pcr": atm_pcr,
        "put_writers_percent": put_writers_percent,
        "call_writers_percent": call_writers_percent,
        "put_unwind_percent": put_unwind_percent,
        "call_unwind_percent": call_unwind_percent,
        "net_bias": net_bias,
        "bias_strength": bias_strength,
        "bias_driver": bias_driver,
        "support": clean(support),
        "support_type": support_type,
        "support_tendency": support_tend,
        "support_oi_pct": sec_put_oi_pct,
        "support_vol_pct": sec_put_vol_pct,
        "support_second_oi_strike": clean(sec_put_oi_strike),
        "support_second_vol_strike": clean(sec_put_vol_strike),
        "resistance": clean(resistance),
        "resistance_type": resistance_type,
        "resistance_tendency": resistance_tend,
        "resistance_oi_pct": sec_call_oi_pct,
        "resistance_vol_pct": sec_call_vol_pct,
        "resistance_second_oi_strike": clean(sec_call_oi_strike),
        "resistance_second_vol_strike": clean(sec_call_vol_strike),
        "eor": eor,
        "eor_plus_1": eor_plus_1,
        "eor_minus_1": eor_minus_1,
        "eos": eos,
        "eos_plus_1": eos_plus_1,
        "eos_minus_1": eos_minus_1,
        "max_pain": clean(max_pain),
        "predictor_signal": predictor_signal,
        "predictor_confidence": predictor_confidence,
        "predictor_reason": predictor_reason,
        "bull_score": bull_score,
        "bear_score": bear_score,
        "smart_level_name": nearest_name,
        "smart_level_value": nearest_value,
        "smart_level_dist": int(round(nearest_value - spot)),
        "smart_conf": conf,
        "smart_break": brk,
        "smart_action": action,
        "trade_mode": trade_mode,
        "spot_confirm": spot_confirm,
        "entry_conf": entry_conf,
        "rows": rows_json
    }

# =====================================================
# ROUTES
# =====================================================
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/data")
def api_data():
    data = fetch_option_chain()

    if data:
        save_history(data)
        data.update(get_db_trends())

    return jsonify(data)


@app.route("/api/history")
def api_history():
    con = sqlite3.connect(DB_FILE)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    cur.execute("SELECT MAX(trade_date) FROM history")
    row = cur.fetchone()

    latest_date = row[0] if row and row[0] else \
        datetime.now().strftime("%Y-%m-%d")

    cur.execute("""
        SELECT *
        FROM history
        WHERE trade_date=?
        ORDER BY trade_time
    """, (latest_date,))

    rows = [dict(x) for x in cur.fetchall()]

    con.close()

    return jsonify(rows)

@app.route("/api/signals")
def api_signals():

    con = sqlite3.connect(DB_FILE)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    cur.execute("""
        SELECT signal, trade_date, trade_time
        FROM signal_log
        ORDER BY trade_date DESC, trade_time DESC
        LIMIT 300
    """)

    rows = [dict(r) for r in cur.fetchall()]
    con.close()

    return jsonify(rows)


@app.route("/api/export_csv")
def export_csv():
    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()

    cur.execute("SELECT MAX(trade_date) FROM history")
    row = cur.fetchone()

    latest_date = row[0] if row and row[0] else \
        datetime.now().strftime("%Y-%m-%d")

    cur.execute("""
        SELECT *
        FROM history
        WHERE trade_date=?
        ORDER BY trade_time
    """, (latest_date,))

    rows = cur.fetchall()
    con.close()

    path = "nima_today_history.csv"

    with open(path, "w", newline="") as f:
        w = csv.writer(f)

        w.writerow([
            "trade_date",
            "trade_time",
            "spot",
            "eor",
            "eos",
            "eor_plus_1",
            "eor_minus_1",
            "eos_plus_1",
            "eos_minus_1",
            "market_pcr",
            "put_writers",
            "call_writers"
        ])

        w.writerows(rows)

    return send_file(path, as_attachment=True)


# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    app.run(debug=True)