# app.py
# Streamlit + Strava coaching dashboard met activity details, streams, zones en rijkere metrics.
# - Haalt recente activiteiten op (paginatie)
# - Optioneel: per-activiteit details (/activities/{id})
# - Optioneel: streams (/activities/{id}/streams) voor tijd-in-zones, TRIMP, pacing-variatie, cadans, heuvelprofiel
# - Haalt athlete HR-zones op (/athlete/zones) en gebruikt die voor zone-analyse
# - Berekent weekvolume, monotony, strain, acute:chronic ratio
# - Haalt gear-informatie op voor schoen-kilometers
# - Toont background.txt en gebruikt de (geüpdatete) tekst in het coach-advies
# - Coach-advies via OpenAI als OPENAI_API_KEY aanwezig is; anders toon ik het samengestelde prompt

import os
import time
import json
import math
import textwrap
from datetime import datetime, timedelta, timezone, date
from typing import Dict, List, Tuple, Optional

import requests
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

# ========= Config =========
STRAVA_API = "https://www.strava.com/api/v3"
TIMEZONE = timezone.utc  # Strava timestamps zijn UTC; voor tonen converteren we naar local met pandas

# Rate-limit vriendelijk
API_SLEEP_S = 0.2

# ========== Helper functies ==========
def api_get(url: str, access_token: str, params: dict = None) -> requests.Response:
    headers = {"Authorization": f"Bearer {access_token}"}
    resp = requests.get(url, headers=headers, params=params or {}, timeout=30)
    if resp.status_code == 401:
        raise RuntimeError("401 Unauthorized. Je access token is ongeldig of verlopen.")
    if resp.status_code == 429:
        raise RuntimeError("429 Rate limit van Strava bereikt. Probeer later opnieuw.")
    resp.raise_for_status()
    return resp

def paginate_activities(access_token: str, after_ts: Optional[int], before_ts: Optional[int], max_pages: int = 5, per_page: int = 50) -> List[dict]:
    """/athlete/activities met paginatie"""
    all_items = []
    for page in range(1, max_pages + 1):
        params = {"per_page": per_page, "page": page}
        if after_ts: params["after"] = after_ts
        if before_ts: params["before"] = before_ts
        r = api_get(f"{STRAVA_API}/athlete/activities", access_token, params)
        items = r.json()
        all_items.extend(items)
        if len(items) < per_page:
            break
        time.sleep(API_SLEEP_S)
    return all_items

def get_activity_detail(access_token: str, activity_id: int) -> dict:
    r = api_get(f"{STRAVA_API}/activities/{activity_id}", access_token, params={"include_all_efforts": "true"})
    return r.json()

def get_activity_streams(access_token: str, activity_id: int, types: List[str]) -> dict:
    params = {"keys": ",".join(types), "key_by_type": "true"}
    r = api_get(f"{STRAVA_API}/activities/{activity_id}/streams", access_token, params=params)
    return r.json()

def get_athlete_zones(access_token: str) -> dict:
    r = api_get(f"{STRAVA_API}/athlete/zones", access_token)
    return r.json()

def get_gear(access_token: str, gear_id: str) -> dict:
    r = api_get(f"{STRAVA_API}/gear/{gear_id}", access_token)
    return r.json()

def safe_pace_s_per_km(distance_m, moving_time_s):
    if not distance_m or distance_m <= 0 or not moving_time_s or moving_time_s <= 0:
        return None
    return moving_time_s / (distance_m / 1000.0)

def fmt_pace(s_per_km: Optional[float]) -> str:
    if s_per_km is None or not np.isfinite(s_per_km):
        return "-"
    m = int(s_per_km // 60)
    s = int(round(s_per_km - m * 60))
    return f"{m}:{s:02d} /km"

def week_start(d: pd.Timestamp) -> pd.Timestamp:
    # Maandag als start (ISO week)
    return d - pd.to_timedelta(d.weekday, unit="D")

def compute_zone_minutes(hr_series: np.ndarray, time_series: np.ndarray, zones: List[Tuple[int, int]]) -> List[float]:
    """
    zones: lijst van (min_incl, max_incl) hartslagzones.
    hr_series: bpm per sample
    time_series: seconden vanaf start per sample
    Return: minuten per zone
    """
    if hr_series is None or time_series is None or len(hr_series) == 0:
        return [0.0] * len(zones)
    # benader dt per sample
    dt = np.diff(time_series, prepend=time_series[0])
    dt[dt < 0] = 0
    mins_per_zone = []
    for lo, hi in zones:
        mask = (hr_series >= lo) & (hr_series <= hi)
        mins = dt[mask].sum() / 60.0
        mins_per_zone.append(float(mins))
    return mins_per_zone

def trimp_from_zone_minutes(zone_minutes: List[float], weights: List[float]) -> float:
    return float(sum(m * w for m, w in zip(zone_minutes, weights)))

def rolling_weeks(df_runs: pd.DataFrame) -> pd.DataFrame:
    # Weekly stats op basis van start_date (local)
    if df_runs.empty:
        return pd.DataFrame()
    g = df_runs.groupby("week_start", as_index=False).agg(
        week_km=("distance_km", "sum"),
        week_time_h=("moving_time", lambda s: s.sum() / 3600.0),
        week_elev=("total_elevation_gain", "sum"),
        n_runs=("id", "count"),
        trimp=("trimp", "sum")
    )
    # Monotony = mean / std (soms gedef. als volume / std; we gebruiken km)
    if len(g) >= 2:
        mu = g["week_km"].mean()
        sd = g["week_km"].std(ddof=0)
        g["monotony"] = (mu / sd) if sd and sd > 0 else np.nan
        g["strain"] = g["monotony"] * g["week_km"]
    else:
        g["monotony"] = np.nan
        g["strain"] = np.nan
    # Acute:Chronic ratio (4w / 12w)
    g["acr"] = np.nan
    if len(g) >= 12:
        g = g.sort_values("week_start")
        g["rolling_4w"] = g["week_km"].rolling(4, min_periods=1).mean()
        g["rolling_12w"] = g["week_km"].rolling(12, min_periods=1).mean()
        g["acr"] = g["rolling_4w"] / g["rolling_12w"]
    return g

def read_background_file() -> str:
    path = "background.txt"
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return ""
    return ""

def write_background_file(text: str):
    try:
        with open("background.txt", "w", encoding="utf-8") as f:
            f.write(text)
    except Exception:
        pass

def openai_advice(prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return f"⚠️ OPENAI_API_KEY niet gevonden. Hieronder zie je het prompt dat anders naar ChatGPT zou gaan:\n\n{prompt}"
    try:
        # v4+ style (openai>=1.0), pas aan indien je client anders is
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Je bent een ervaren hardloopcoach, gespecialiseerd in marathonvoorbereiding."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            max_tokens=1200,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"⚠️ Fout bij OpenAI call: {e}\n\nPrompt was:\n{prompt}"

def localize(ts_str: str) -> pd.Timestamp:
    # Strava geeft start_date_local en start_date (UTC). We prefer start_date_local als string.
    try:
        return pd.to_datetime(ts_str)
    except Exception:
        return pd.NaT

# ========== UI ==========
st.set_page_config(page_title="Strava Coach Dashboard", layout="wide")
st.title("🏃 Strava Coach Dashboard — NY Marathon")

with st.sidebar:
    st.header("🔐 Strava toegang")
    default_token = os.getenv("STRAVA_ACCESS_TOKEN", "")
    access_token = st.text_input("Access token (Bearer)", value=default_token, type="password")
    st.caption("Tip: voeg STRAVA_ACCESS_TOKEN toe als environment variabele voor automatische invulling.")

    st.header("📅 Periode & opties")
    # Range slider in weken
    weeks_back = st.slider("Aantal weken terug (analysevenster)", min_value=4, max_value=26, value=12, step=1)

    opt_details = st.toggle("Per-activiteit details ophalen (splits, best efforts, cadans…)", value=True)
    opt_streams = st.toggle("Streams analyseren voor laatste N runs", value=True)
    n_streams = st.number_input("Aantal runs met streams", min_value=5, max_value=50, value=12, step=1)

    st.header("📈 Zones & load")
    # Zone gewichten voor TRIMP (bijv. 1,2,3,4,5)
    zone_weights_str = st.text_input("Zone-gewichten (TRIMP), komma-gescheiden", value="1,2,3,4,5")

    st.header("💾 Output")
    show_prompt = st.toggle("Toon volledige coach-prompt bij output", value=False)

if not access_token:
    st.warning("Voer je Strava access token in om te starten.")
    st.stop()

# ========== Data ophalen ==========
today = datetime.now(timezone.utc)
after = today - timedelta(weeks=int(weeks_back))
after_ts = int(after.timestamp())
before_ts = int(today.timestamp())

with st.spinner("🏗️ Activiteiten laden…"):
    raw_acts = paginate_activities(access_token, after_ts=after_ts, before_ts=before_ts, max_pages=5, per_page=50)

if not raw_acts:
    st.info("Geen activiteiten gevonden in de gekozen periode.")
    st.stop()

# DataFrame basis
df = pd.json_normalize(raw_acts)
# Filter alleen runs
if "sport_type" in df.columns:
    df = df[df["sport_type"] == "Run"].copy()
if df.empty:
    st.info("Geen hardloopactiviteiten gevonden in deze periode.")
    st.stop()

# Basis kolommen
df["start_local"] = pd.to_datetime(df.get("start_date_local", df.get("start_date")))
df["distance_km"] = df["distance"] / 1000.0
df["pace_s_per_km"] = df.apply(lambda r: safe_pace_s_per_km(r.get("distance"), r.get("moving_time")), axis=1)
df["pace"] = df["pace_s_per_km"].map(fmt_pace)
df["week_start"] = df["start_local"].dt.tz_localize(None).map(week_start)

basic_cols = [
    "id", "name", "start_local", "distance_km", "moving_time", "pace",
    "average_heartrate", "total_elevation_gain", "has_heartrate", "gear_id"
]
present_cols = [c for c in basic_cols if c in df.columns]
df_basic = df[present_cols].sort_values("start_local", ascending=False).reset_index(drop=True)

# ========== Athlete zones ophalen ==========
zones_json = {}
zone_bounds = []   # list of (lo, hi)
zone_labels = []   # Z1..Z5
try:
    zones_json = get_athlete_zones(access_token)
    # Verwacht format: {"heart_rate": {"custom_zones": bool, "zones":[{"min":..,"max":..},...]}}
    z = (zones_json or {}).get("heart_rate", {}).get("zones", [])
    for i, zz in enumerate(z, start=1):
        lo = int(zz.get("min", 0))
        hi = int(zz.get("max", lo))
        zone_bounds.append((lo, hi))
        zone_labels.append(f"Z{i}")
except Exception as e:
    st.warning(f"Kon HR-zones niet ophalen ({e}). Ik schat zones later indien nodig.")

# fallback wanneer geen zones
if not zone_bounds:
    # eenvoudige 5-zone schatting t.o.v. maxHR=190
    max_hr = 190
    bps = [0.6, 0.7, 0.8, 0.9, 1.0]
    lows = [int(max_hr * 0.5), int(max_hr * 0.6), int(max_hr * 0.7), int(max_hr * 0.8), int(max_hr * 0.9)]
    highs = [int(max_hr * p) for p in bps]
    zone_bounds = list(zip(lows, highs))
    zone_labels = [f"Z{i}" for i in range(1, 6)]

# TRIMP-gewichten
try:
    zone_weights = [float(x.strip()) for x in zone_weights_str.split(",") if x.strip()]
except Exception:
    zone_weights = [1, 2, 3, 4, 5]
if len(zone_weights) != len(zone_bounds):
    # pad / truncate
    if len(zone_weights) < len(zone_bounds):
        zone_weights = zone_weights + [zone_weights[-1]] * (len(zone_bounds) - len(zone_weights))
    else:
        zone_weights = zone_weights[:len(zone_bounds)]

# ========== Details ophalen ==========
details_records = []
if opt_details:
    with st.spinner("📄 Activiteit-details ophalen…"):
        for aid in df_basic["id"].tolist():
            try:
                d = get_activity_detail(access_token, int(aid))
                keep = {
                    "id": d.get("id"),
                    "average_cadence": d.get("average_cadence"),
                    "device_watts": d.get("device_watts"),
                    "average_stride_length": d.get("average_stride_length"),
                    "max_heartrate": d.get("max_heartrate"),
                    "suffer_score": d.get("suffer_score"),  # legacy bij sommigen
                    "has_heartrate": d.get("has_heartrate"),
                    "best_efforts": d.get("best_efforts", []),
                    "splits_metric": d.get("splits_metric", []),
                    "laps": d.get("laps", []),
                    "segment_efforts": d.get("segment_efforts", []),
                }
                details_records.append(keep)
            except Exception as e:
                details_records.append({"id": aid})
            time.sleep(API_SLEEP_S)

details_df = pd.DataFrame(details_records) if details_records else pd.DataFrame(columns=["id"])

# Merge basis + details
df_merged = df_basic.merge(details_df, on="id", how="left")

# ========== Streams ophalen voor laatste N runs ==========
stream_types = ["time", "distance", "heartrate", "velocity_smooth", "cadence", "grade_smooth"]
streams_stats = []  # per activiteit: zone-minuten, trimp, variabiliteitsmetrics
if opt_streams:
    with st.spinner("📊 Streams analyseren…"):
        ids_for_streams = df_merged["id"].tolist()[: int(n_streams)]
        for aid in ids_for_streams:
            try:
                sjson = get_activity_streams(access_token, int(aid), stream_types)
                t = np.array(sjson.get("time", {}).get("data", []), dtype=float) if "time" in sjson else None
                hr = np.array(sjson.get("heartrate", {}).get("data", []), dtype=float) if "heartrate" in sjson else None
                vel = np.array(sjson.get("velocity_smooth", {}).get("data", []), dtype=float) if "velocity_smooth" in sjson else None
                cad = np.array(sjson.get("cadence", {}).get("data", []), dtype=float) if "cadence" in sjson else None
                grd = np.array(sjson.get("grade_smooth", {}).get("data", []), dtype=float) if "grade_smooth" in sjson else None

                zone_mins = compute_zone_minutes(hr, t, zone_bounds) if hr is not None and t is not None else [0.0]*len(zone_bounds)
                trimp = trimp_from_zone_minutes(zone_mins, zone_weights)

                # Variabiliteit (coëfficiënt of variation) van tempo en cadans
                pace_s = None
                if vel is not None and len(vel) > 0:
                    # vel in m/s -> pace s/km
                    with np.errstate(divide='ignore', invalid='ignore'):
                        pace_s = np.where(vel > 0, 1000.0 / vel, np.nan)
                    pace_cv = float(np.nanstd(pace_s) / np.nanmean(pace_s)) if np.nanmean(pace_s) and np.nanmean(pace_s) > 0 else np.nan
                else:
                    pace_cv = np.nan

                cad_cv = float(np.nanstd(cad) / np.nanmean(cad)) if cad is not None and np.nanmean(cad) else np.nan
                hill_frac = float(np.mean(np.abs(grd) >= 3.0)) if grd is not None and len(grd) else np.nan  # % samples met |gradient|>=3%

                streams_stats.append({
                    "id": aid,
                    **{f"{lbl}_min": m for lbl, m in zip(zone_labels, zone_mins)},
                    "trimp": trimp,
                    "pace_cv": pace_cv,
                    "cadence_cv": cad_cv,
                    "hill_fraction": hill_frac
                })
            except Exception:
                streams_stats.append({"id": aid})
            time.sleep(API_SLEEP_S)

streams_df = pd.DataFrame(streams_stats) if streams_stats else pd.DataFrame(columns=["id"])
df_merged = df_merged.merge(streams_df, on="id", how="left")

# TRIMP fallback voor runs zonder streams: benader met HR gemiddelde indien beschikbaar
def approx_trimp(row) -> float:
    if pd.notna(row.get("trimp")):
        return row["trimp"]
    avg_hr = row.get("average_heartrate")
    mv_min = (row.get("moving_time") or 0) / 60.0
    if pd.isna(avg_hr) or not mv_min or mv_min <= 0:
        return 0.0
    # approximatie: plaats avg_hr in dichtsbijzijnde zone
    z_idx = 0
    for i, (lo, hi) in enumerate(zone_bounds):
        if lo <= avg_hr <= hi:
            z_idx = i
            break
    w = zone_weights[z_idx] if z_idx < len(zone_weights) else zone_weights[-1]
    return float(mv_min * w)

if "trimp" not in df_merged.columns:
    df_merged["trimp"] = np.nan
df_merged["trimp"] = df_merged.apply(approx_trimp, axis=1)

# Weekmetrics
weekly = rolling_weeks(df_merged)

# Gear (schoenen)
gear_map = {}
if "gear_id" in df_merged.columns and df_merged["gear_id"].notna().any():
    with st.spinner("👟 Schoen-gegevens ophalen…"):
        for gid in sorted(df_merged["gear_id"].dropna().unique().tolist()):
            try:
                gj = get_gear(access_token, gid)
                gear_map[gid] = {
                    "name": gj.get("name"),
                    "brand_name": gj.get("brand_name"),
                    "distance_km": (gj.get("distance", 0) or 0) / 1000.0
                }
                time.sleep(API_SLEEP_S)
            except Exception:
                pass

# ========== UI weergave ==========
st.subheader("Overzicht activiteiten")
st.caption("Recente hardloopactiviteiten in de gekozen periode.")
st.dataframe(
    df_merged[["id","name","start_local","distance_km","moving_time","pace","average_heartrate","total_elevation_gain","average_cadence","Z1_min","Z2_min","Z3_min","Z4_min","Z5_min","trimp"]].fillna(""),
    use_container_width=True
)

# Weekgrafieken
st.subheader("Weekoverzicht")
if not weekly.empty:
    w1 = alt.Chart(weekly).mark_bar().encode(
        x=alt.X("week_start:T", title="Week"),
        y=alt.Y("week_km:Q", title="Kilometers per week"),
        tooltip=["week_start","week_km","n_runs","week_time_h","week_elev","trimp","monotony","strain","acr"]
    ).properties(height=220)
    st.altair_chart(w1, use_container_width=True)

    w2 = alt.Chart(weekly).mark_line(point=True).encode(
        x=alt.X("week_start:T", title="Week"),
        y=alt.Y("trimp:Q", title="TRIMP per week"),
        tooltip=["week_start","trimp","n_runs"]
    ).properties(height=220)
    st.altair_chart(w2, use_container_width=True)
else:
    st.info("Onvoldoende data voor weekoverzicht.")

# Zones samenvatting (laatste N geanalyseerde runs)
if opt_streams and not streams_df.empty:
    st.subheader(f"Tijd in zones (laatste {int(n_streams)} runs)")
    zone_cols = [f"{z}_min" for z in zone_labels]
    zsum = df_merged[zone_cols].sum(numeric_only=True)
    zdf = pd.DataFrame({"zone": zone_labels, "minuten": [zsum.get(c, 0.0) for c in zone_cols]})
    zchart = alt.Chart(zdf).mark_bar().encode(
        x=alt.X("zone:N", title="Zone"),
        y=alt.Y("minuten:Q", title="Min."),
        tooltip=["zone","minuten"]
    ).properties(height=220)
    st.altair_chart(zchart, use_container_width=True)

# Schoenen
if gear_map:
    st.subheader("Schoen-kilometers (totaal volgens Strava)")
    gdf = pd.DataFrame([
        {"gear_id": gid, "naam": val["name"], "merk": val["brand_name"], "totaal_km": val["distance_km"]}
        for gid, val in gear_map.items()
    ])
    st.dataframe(gdf, use_container_width=True)

# ========== Samenvattingen ==========
def agg_period(df_in: pd.DataFrame, weeks: int) -> dict:
    cutoff = pd.Timestamp(datetime.now()).tz_localize(None) - pd.Timedelta(weeks=weeks)
    s = df_in[df_in["start_local"] >= cutoff]
    if s.empty:
        return {"km": 0, "n_runs": 0, "avg_pace": None, "avg_hr": None, "elev": 0, "trimp": 0}
    km = float(s["distance_km"].sum())
    n = int(s["id"].count())
    avg_pace = fmt_pace(np.nanmean(s["pace_s_per_km"])) if "pace_s_per_km" in s else None
    avg_hr = float(np.nanmean(s["average_heartrate"])) if "average_heartrate" in s else None
    elev = float(s["total_elevation_gain"].sum())
    trimp_sum = float(s["trimp"].sum())
    long_run_km = float(s["distance_km"].max())
    long_run_time = float(s["moving_time"].max())
    return {
        "km": km, "n_runs": n, "avg_pace": avg_pace, "avg_hr": avg_hr,
        "elev": elev, "trimp": trimp_sum,
        "long_run_km": long_run_km, "long_run_time_min": long_run_time/60.0
    }

sum_4w = agg_period(df_merged, 4)
sum_12w = agg_period(df_merged, 12)

st.subheader("Samenvatting 4 & 12 weken")
c1, c2 = st.columns(2)
with c1:
    st.metric("4w km", f"{sum_4w['km']:.1f}")
    st.metric("4w runs", f"{sum_4w['n_runs']}")
    st.metric("4w TRIMP", f"{sum_4w['trimp']:.0f}")
    st.write(f"Gem. pace: {sum_4w['avg_pace'] or '-'} — Gem. HR: {sum_4w['avg_hr']:.0f}" if sum_4w['avg_hr'] else f"Gem. pace: {sum_4w['avg_pace'] or '-'}")
    st.write(f"Langste duurloop: {sum_4w['long_run_km']:.1f} km (~{sum_4w['long_run_time_min']:.0f} min)")
with c2:
    st.metric("12w km", f"{sum_12w['km']:.1f}")
    st.metric("12w runs", f"{sum_12w['n_runs']}")
    st.metric("12w TRIMP", f"{sum_12w['trimp']:.0f}")
    st.write(f"Gem. pace: {sum_12w['avg_pace'] or '-'} — Gem. HR: {sum_12w['avg_hr']:.0f}" if sum_12w['avg_hr'] else f"Gem. pace: {sum_12w['avg_pace'] or '-'}")
    st.write(f"Langste duurloop: {sum_12w['long_run_km']:.1f} km (~{sum_12w['long_run_time_min']:.0f} min)")

# ========= Achtergrondtekst & Coach-advies =========
st.subheader("Achtergrondinformatie (background.txt)")
bg_text = st.text_area("Bewerk je achtergrondcontext voor het advies:", value=read_background_file(), height=220)
colb1, colb2 = st.columns([1,1])
with colb1:
    if st.button("💾 Sla achtergrond op"):
        write_background_file(bg_text)
        st.success("background.txt opgeslagen.")

# Coach prompt samenstellen
def build_coach_prompt() -> str:
    profile_bits = []
    # Basic profiel: laatste 4/12 weken + zones info
    profile_bits.append("== Samenvattingen ==")
    profile_bits.append(f"4w: km={sum_4w['km']:.1f}, runs={sum_4w['n_runs']}, TRIMP={sum_4w['trimp']:.0f}, avg_pace={sum_4w['avg_pace']}, avg_hr={sum_4w['avg_hr']}, long_run_km={sum_4w['long_run_km']:.1f}")
    profile_bits.append(f"12w: km={sum_12w['km']:.1f}, runs={sum_12w['n_runs']}, TRIMP={sum_12w['trimp']:.0f}, avg_pace={sum_12w['avg_pace']}, avg_hr={sum_12w['avg_hr']}, long_run_km={sum_12w['long_run_km']:.1f}")

    if opt_streams and not streams_df.empty:
        zsum = df_merged[[f"{z}_min" for z in zone_labels if f"{z}_min" in df_merged.columns]].sum(numeric_only=True)
        profile_bits.append("== Tijd in zones (laatste N runs) in minuten ==")
        for z in zone_labels:
            key = f"{z}_min"
            if key in zsum:
                profile_bits.append(f"{z}: {float(zsum[key]):.1f} min")

    if not weekly.empty and "acr" in weekly.columns:
        last_row = weekly.sort_values("week_start").iloc[-1]
        profile_bits.append("== Weekbelasting ==")
        profile_bits.append(f"Huidige week km: {last_row['week_km']:.1f}, monotony: {last_row['monotony'] if pd.notna(last_row['monotony']) else 'NA'}, strain: {last_row['strain'] if pd.notna(last_row['strain']) else 'NA'}, A:C ratio: {last_row['acr'] if pd.notna(last_row['acr']) else 'NA'}")

    if gear_map:
        profile_bits.append("== Schoenen totaal volgens Strava ==")
        for gid, val in gear_map.items():
            profile_bits.append(f"- {val['brand_name']} {val['name']}: {val['distance_km']:.0f} km")

    # Prompt
    prompt = f"""
Context (door gebruiker bewerkbaar):
{bg_text}

Profieldata uit Strava:
{chr(10).join(profile_bits)}

Wat je moet doen als coach:
- Geef een concreet 2-weeks microplan met sessies per dag (in zones Z1–Z5) afgestemd op de data.
- Benoem volume- en intensiteitsdoelen, en licht toe waarom (op basis van tijd-in-zones, long run, trimp, monotony/strain).
- Geef aanpassingen als A:C ratio > 1.5 of monotony > 2.0 (waarschuwing voor overbelasting).
- Voeg tips toe voor heuvelwerk en cadans als hill_fraction of cadence_cv dat suggereert.
- Eindig met 3 bullets “Let hierop”.
Schrijf kort, puntsgewijs en in het Nederlands.
""".strip()
    return prompt

with colb2:
    if st.button("🧠 Genereer coach-advies"):
        prompt = build_coach_prompt()
        if show_prompt:
            with st.expander("Coach-prompt (debug)"):
                st.code(prompt)
        advice = openai_advice(prompt)
        st.markdown("### ✍️ Coach-advies")
        st.write(advice)

# ========== Extra visuals ==========
st.subheader("Variabiliteit & heuvelprofiel (laatste N runs)")
var_cols = ["pace_cv", "cadence_cv", "hill_fraction"]
vdf = df_merged[["start_local","name"] + [c for c in var_cols if c in df_merged.columns]].dropna(how="all", subset=var_cols)
if not vdf.empty and any(c in vdf.columns for c in var_cols):
    for metric in var_cols:
        if metric in vdf.columns and vdf[metric].notna().any():
            chart = alt.Chart(vdf).mark_line(point=True).encode(
                x=alt.X("start_local:T", title="Datum"),
                y=alt.Y(f"{metric}:Q", title=metric),
                tooltip=["name","start_local",metric]
            ).properties(height=200)
            st.altair_chart(chart, use_container_width=True)
else:
    st.caption("Geen stream-variabiliteit beschikbaar voor de geselecteerde runs.")

st.success("Klaar! Gebruik de toggles in de sidebar om meer/anders te analyseren.")
