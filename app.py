import os
import time
import json
import math
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from dateutil import tz
import streamlit as st

# =========================
# Config
# =========================
STRAVA_CLIENT_ID = st.secrets["STRAVA_CLIENT_ID"]
STRAVA_CLIENT_SECRET = st.secrets["STRAVA_CLIENT_SECRET"]
STRAVA_REFRESH_TOKEN = st.secrets["STRAVA_REFRESH_TOKEN"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
OPENAI_MODEL = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")

STRAVA_TOKEN_URL = "https://www.strava.com/oauth/token"
STRAVA_API_BASE = "https://www.strava.com/api/v3"

st.set_page_config(page_title="NY Marathon Coach", page_icon="🏃‍♂️", layout="wide")
st.title("🏃‍♂️ NY Marathon Coach — Strava → Persoonlijk advies")

# =========================
# Helpers: Strava OAuth
# =========================
@st.cache_data(ttl=50*60)  # cache ~50min; access tokens zijn ±6 uur geldig, maar dit houdt API-lasten laag
def get_access_token(refresh_token: str) -> dict:
    """Vraag met refresh_token een verse access_token op."""
    resp = requests.post(STRAVA_TOKEN_URL, data={
        "client_id": STRAVA_CLIENT_ID,
        "client_secret": STRAVA_CLIENT_SECRET,
        "grant_type": "refresh_token",
        "refresh_token": refresh_token
    }, timeout=30)
    resp.raise_for_status()
    return resp.json()  # bevat access_token, expires_at, athlete, etc.

def bearer_header(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}

# =========================
# Helpers: Strava data ophalen
# =========================
def fetch_activities(access_token: str, after: datetime | None = None, per_page: int = 100) -> pd.DataFrame:
    """Haal recente activiteiten op (hardlopen / alles) en geef DataFrame terug."""
    all_rows = []
    page = 1
    params = {"per_page": per_page, "page": page}
    if after:
        # Strava verwacht Unix epoch seconds (UTC)
        params["after"] = int(after.replace(tzinfo=timezone.utc).timestamp())

    while True:
        params["page"] = page
        r = requests.get(f"{STRAVA_API_BASE}/athlete/activities", headers=bearer_header(access_token), params=params, timeout=30)
        r.raise_for_status()
        chunk = r.json()
        if not chunk:
            break
        all_rows.extend(chunk)
        if len(chunk) < per_page:
            break
        page += 1

    if not all_rows:
        return pd.DataFrame()

    df = pd.json_normalize(all_rows)
    # nuttige kolommen selecteren
    cols = [
        "name","type","sport_type","start_date_local","distance","moving_time","elapsed_time",
        "total_elevation_gain","average_speed","max_speed","average_heartrate","max_heartrate",
        "has_heartrate","suffer_score","private","visibility","kudos_count"
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df[cols].copy()

    # conversies
    df["start_date_local"] = pd.to_datetime(df["start_date_local"])
    df["distance_km"] = df["distance"].fillna(0) / 1000.0
    df["pace_min_per_km"] = df["moving_time"] / (df["distance_km"] * 60)
    # nette tijd string
    def fmt_pace(x):
        if x is None or pd.isna(x) or not math.isfinite(x) or x<=0:
            return None
        m = int(x)
        s = int(round((x-m)*60))
        return f"{m}:{s:02d} /km"
    df["pace"] = df["pace_min_per_km"].apply(fmt_pace)
    return df.sort_values("start_date_local", ascending=False)

# =========================
# UI: parameters
# =========================
col1, col2, col3 = st.columns([1,1,2])
with col1:
    weeks_back = st.slider("Hoeveel weken terug ophalen?", 4, 26, 12)
with col2:
    marathon_date = st.date_input("Marathon datum", value=datetime(datetime.now().year+1, 11, 1).date(),
                                  help="Gebruik de exacte datum van de NYC Marathon die jij gaat lopen.")
with col3:
    goal = st.selectbox("Doel", ["Uitlopen", "PR lopen", "Tijdsspecifiek"], index=0)
    target_time = st.text_input("Streeftijd (hh:mm:ss)", value="", help="Alleen invullen bij Tijdsspecifiek.")

after_dt = datetime.now(timezone.utc) - timedelta(weeks=weeks_back)

# =========================
# Data ophalen
# =========================
with st.spinner("Tokens verversen en activiteiten ophalen..."):
    token_payload = get_access_token(STRAVA_REFRESH_TOKEN)
    access_token = token_payload["access_token"]
    activities = fetch_activities(access_token, after=after_dt)

if activities.empty:
    st.warning("Geen activiteiten gevonden in de gekozen periode. Probeer meer weken terug.")
    st.stop()

st.subheader("📋 Recente activiteiten")
st.dataframe(activities[["start_date_local","name","sport_type","distance_km","pace","average_heartrate","total_elevation_gain"]], use_container_width=True)

# =========================
# Samenvatting voor advies
# =========================
def summarize_block(df: pd.DataFrame, weeks: int = 4) -> dict:
    cutoff = datetime.now(tz=timezone.utc) - timedelta(weeks=weeks)
    d = df[df["start_date_local"] >= cutoff]
    if d.empty:
        return {"weeks": weeks, "km": 0, "longest_km": 0, "runs": 0, "avg_pace": None, "avg_hr": None, "elev": 0}
    km = float(d["distance_km"].sum())
    longest = float(d["distance_km"].max())
    runs = int((d["sport_type"] == "Run").sum()) if "sport_type" in d.columns else len(d)
    avg_pace = d["pace_min_per_km"].replace([pd.NA, pd.NaT], float("nan")).dropna()
    avg_pace = float(avg_pace.mean()) if len(avg_pace) else None
    avg_hr = d["average_heartrate"].dropna().mean() if "average_heartrate" in d.columns else None
    elev = float(d["total_elevation_gain"].dropna().sum())
    return {"weeks": weeks, "km": round(km,1), "longest_km": round(longest,1), "runs": runs,
            "avg_pace": avg_pace, "avg_hr": round(avg_hr,1) if avg_hr else None, "elev": round(elev,0)}

last4 = s
