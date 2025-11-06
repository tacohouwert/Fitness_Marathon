# app.py
import os
import json
import math
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st
from openai import OpenAI

# =========[ Streamlit basis ]=========
st.set_page_config(
    page_title="NY Marathon Coach",
    page_icon="🏃‍♂️",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.title("🏃‍♂️ NY Marathon Coach — Strava → Persoonlijk advies")

# =========[ Config uit Secrets met nette foutafhandeling ]=========
REQUIRED_SECRETS = [
    "STRAVA_CLIENT_ID",
    "STRAVA_CLIENT_SECRET",
    "STRAVA_REFRESH_TOKEN",
    "OPENAI_API_KEY",
]
missing = [k for k in REQUIRED_SECRETS if k not in st.secrets]
if missing:
    st.error(
        "❌ Ontbrekende secrets: " + ", ".join(missing) +
        "\n\nGa in **Streamlit Cloud → Manage app → Settings → Secrets** en voeg de volgende regels toe:\n\n"
        "```toml\n"
        "STRAVA_CLIENT_ID = \"...\"\n"
        "STRAVA_CLIENT_SECRET = \"...\"\n"
        "STRAVA_REFRESH_TOKEN = \"...\"\n"
        "OPENAI_API_KEY = \"sk-...\"\n"
        "OPENAI_MODEL = \"gpt-4o-mini\"  # optioneel\n"
        "```"
    )
    st.stop()

STRAVA_CLIENT_ID = st.secrets["STRAVA_CLIENT_ID"]
STRAVA_CLIENT_SECRET = st.secrets["STRAVA_CLIENT_SECRET"]
STRAVA_REFRESH_TOKEN = st.secrets["STRAVA_REFRESH_TOKEN"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
OPENAI_MODEL = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")

STRAVA_TOKEN_URL = "https://www.strava.com/oauth/token"
STRAVA_API_BASE = "https://www.strava.com/api/v3"

client = OpenAI(api_key=OPENAI_API_KEY)

# =========[ Sidebar: Persoonlijk profiel ]=========
with st.sidebar:
    st.header("👤 Persoonlijk profiel")
    pt_days = st.multiselect(
        "PT-dagen",
        ["maandag", "dinsdag", "woensdag", "donderdag", "vrijdag"],
        default=["maandag", "woensdag"],
    )
    pt_times = st.text_input("PT tijden (bijv. ma 07:30, wo 18:00)", value="ma 20:00, wo 18:45")
    hockey = st.checkbox("Ik hockey op donderdag", value=True)
    hockey_time = st.text_input("Hockey tijd", value="do 20:00–21:30")
    long_run_day = st.selectbox("Voorkeursdag lange duurloop", ["zaterdag", "zondag", "vrijdag"], index=0)
    rest_days = st.multiselect(
        "Vaste rustdag(en)",
        ["maandag", "dinsdag", "woensdag", "donderdag", "vrijdag", "zaterdag", "zondag"],
        default=[],
    )
    st.markdown("—")
    weekly_availability = st.text_area(
        "Beschikbaarheid (vrije vensters)",
        value="di 07:00–08:00, vr 12:00–13:00",
        placeholder="Bijv. di 07:00–08:00, vr 12:00–13:00",
    )
    other_constraints = st.text_area(
        "Extra context / beperkingen",
        value="Liever geen intensieve intervallen op PT-dagen.",
        placeholder="Blessurehistorie, voorkeuren, reistijd, etc.",
    )
    st.caption("Dit profiel wordt meegenomen in het coach-advies.")

# =========[ Helpers ]=========
def bearer_header(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}

@st.cache_data(ttl=50 * 60)
def get_access_token(refresh_token: str) -> dict:
    resp = requests.post(
        STRAVA_TOKEN_URL,
        data={
            "client_id": STRAVA_CLIENT_ID,
            "client_secret": STRAVA_CLIENT_SECRET,
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()

def _fmt_pace(min_per_km: float | None) -> str | None:
    if min_per_km is None or not math.isfinite(min_per_km) or min_per_km <= 0:
        return None
    m = int(min_per_km)
    s = int(round((min_per_km - m) * 60))
    return f"{m}:{s:02d} /km"

def _safe_mean(a):
    a = [x for x in a if x is not None and math.isfinite(x)]
    return float(np.mean(a)) if a else None

# =========[ Fetch activities (summary list) ]=========
def _normalize_activities(rows: list[dict]) -> pd.DataFrame:
    if not rows: return pd.DataFrame()
    df = pd.json_normalize(rows)
    # zorg dat id aanwezig is en type colums
    if "id" not in df.columns: df["id"] = None
    cols = [
        "id","name","type","sport_type","start_date_local","distance","moving_time","elapsed_time",
        "total_elevation_gain","average_speed","max_speed","average_heartrate","max_heartrate",
        "has_heartrate","suffer_score","private","visibility","kudos_count","gear_id","workout_type"
    ]
    for c in cols:
        if c not in df.columns: df[c] = None
    df = df[cols].copy()
    df["start_date_local"] = pd.to_datetime(df["start_date_local"])
    df["distance_km"] = df["distance"].fillna(0) / 1000.0
    df["pace_min_per_km"] = df["moving_time"] / (df["distance_km"] * 60)
    df["pace"] = df["pace_min_per_km"].apply(_fmt_pace)
    return df.sort_values("start_date_local", ascending=False)

def _fetch_activities_nocache(access_token: str, after_ts: int, per_page: int = 50, max_pages: int = 3) -> pd.DataFrame:
    all_rows = []
    for page in range(1, max_pages + 1):
        r = requests.get(
            f"{STRAVA_API_BASE}/athlete/activities",
            headers=bearer_header(access_token),
            params={"per_page": per_page, "page": page, "after": after_ts},
            timeout=30,
        )
        if r.status_code == 429:
            st.error("Strava API-limiet (429). Probeer later opnieuw.")
            r.raise_for_status()
        r.raise_for_status()
        chunk = r.json()
        if not chunk: break
        all_rows.extend(chunk)
        if len(chunk) < per_page: break
        time.sleep(0.35)
    return _normalize_activities(all_rows)

@st.cache_data(ttl=15 * 60)
def fetch_activities_cached(access_token: str, after_ts: int, per_page: int, max_pages: int) -> pd.DataFrame:
    return _fetch_activities_nocache(access_token, after_ts, per_page, max_pages)

# =========[ Depth: details & streams & zones ]=========
@st.cache_data(ttl=60 * 60)
def fetch_activity_detail(access_token: str, activity_id: int) -> dict:
    r = requests.get(
        f"{STRAVA_API_BASE}/activities/{activity_id}",
        headers=bearer_header(access_token),
        params={"include_all_efforts": "true"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=60 * 60)
def fetch_streams(access_token: str, activity_id: int, keys=("time","distance","velocity_smooth","heartrate","cadence","altitude")) -> dict:
    r = requests.get(
        f"{STRAVA_API_BASE}/activities/{activity_id}/streams",
        headers=bearer_header(access_token),
        params={"keys": ",".join(keys), "key_by_type": "true"},
        timeout=30,
    )
    # 404 kan voorkomen bij oude activiteiten zonder streams → behandel mild
    if r.status_code == 404:
        return {}
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=6 * 60 * 60)
def fetch_zones(access_token: str) -> dict | None:
    r = requests.get(f"{STRAVA_API_BASE}/athlete/zones", headers=bearer_header(access_token), timeout=30)
    if not r.ok: return None
    return r.json()

def hr_decoupling(vel_list, hr_list):
    if not vel_list or not hr_list: return None
    n = min(len(vel_list), len(hr_list))
    if n < 240:  # ~ >=4 min data om ruis te dempen
        return None
    v = np.array(vel_list[:n], dtype=float); hr = np.array(hr_list[:n], dtype=float)
    # filter onrealistische waardes
    hr = np.where(hr <= 0, np.nan, hr); v = np.where(v <= 0, np.nan, v)
    mask = ~np.isnan(hr) & ~np.isnan(v)
    if mask.sum() < 200: return None
    v = v[mask]; hr = hr[mask]
    half = len(v) // 2
    ratio1 = v[:half].mean() / max(hr[:half].mean(), 1e-6)
    ratio2 = v[half:].mean() / max(hr[half:].mean(), 1e-6)
    return (ratio2 - ratio1) / ratio1  # ~ >3% = merkbare drift

def compute_time_in_zones(hr_stream, zones):
    """Return dict {Z1: sec, Z2: sec, ...} op basis van athlete HR zones (custom/auto)."""
    if not hr_stream or not zones: return None
    # zones: zones["heart_rate"]["zones"] met lower/upper
    zdef = zones.get("heart_rate", {}).get("zones", [])
    if not zdef: return None
    counts = [0]*len(zdef)
    for hr in hr_stream:
        if hr is None or hr <= 0: continue
        for i, z in enumerate(zdef):
            lo = z.get("min", z.get("lower", 0))
            hi = z.get("max", z.get("upper", 10**9))
            if lo <= hr <= hi:
                counts[i] += 1  # assumptie: 1 sample ~1s
                break
    # maak labels Z1..Zn
    return {f"Z{i+1}": sec for i, sec in enumerate(counts)}

def z_ratio_low_vs_high(tiz: dict):
    """Return easy vs quality ratio (~Z1-2 vs Z3+)."""
    if not tiz: return None
    easy = int(tiz.get("Z1",0)) + int(tiz.get("Z2",0))
    quality = sum(int(v) for k,v in tiz.items() if k not in ("Z1","Z2"))
    total = easy + quality
    if total == 0: return None
    return easy/total, quality/total

# =========[ UI: hoofdpagina-parameters ]=========
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    weeks_back = st.slider("Hoeveel weken terug ophalen?", 4, 26, 8)
with col2:
    marathon_date = st.date_input(
        "Marathon datum",
        value=datetime(datetime.now().year + 1, 11, 1).date(),
        help="Exacte marathondatum helpt de planning.",
    )
with col3:
    goal = st.selectbox("Doel", ["Uitlopen", "PR lopen", "Tijdsspecifiek"], index=0)
    target_time = st.text_input("Streeftijd (hh:mm:ss)", value="", help="Alleen invullen bij Tijdsspecifiek.")

# 🔬 Diepte-analyse toggle
st.markdown("---")
deep_col1, deep_col2 = st.columns([1,2])
with deep_col1:
    enable_deep = st.checkbox("🔬 Diepte-analyse (meer API-calls)", value=False, help="Details, streams, zones, HR-drift en time-in-zone.")
with deep_col2:
    deep_n = st.slider("Aantal recente runs voor diepte-analyse", min_value=3, max_value=12, value=6, step=1)

# =========[ Background.txt → eerst bewerken, dan genereren ]=========
bg_path = os.path.join(os.getcwd(), "background.txt")
if "background_text" not in st.session_state:
    if os.path.exists(bg_path):
        try:
            with open(bg_path, "r", encoding="utf-8") as f:
                st.session_state.background_text = f.read().strip()
            st.success("Achtergrondinformatie geladen uit background.txt")
        except Exception as e:
            st.session_state.background_text = ""
            st.warning(f"Kon background.txt niet lezen: {e}")
    else:
        st.session_state.background_text = ""
        st.info("Optioneel: plaats een 'background.txt' in de root. Je kunt hieronder direct tekst invullen of aanpassen.")

st.subheader("📝 Achtergrondinformatie (bewerkbaar)")
updated_background = st.text_area(
    "Deze tekst wordt meegestuurd naar het coach-advies.",
    value=st.session_state.background_text,
    height=200,
)
st.caption("Wijzigingen gelden voor deze sessie (niet automatisch opgeslagen).")

# =========[ Data ophalen van Strava ]=========
after_dt = datetime.now(timezone.utc) - timedelta(weeks=weeks_back)
after_ts = int(after_dt.timestamp())

with st.spinner("Tokens verversen en activiteiten ophalen…"):
    token_payload = get_access_token(STRAVA_REFRESH_TOKEN)
    access_token = token_payload["access_token"]
    try:
        activities = fetch_activities_cached(access_token, after_ts, per_page=50, max_pages=3)
    except requests.HTTPError:
        st.stop()

if activities.empty:
    st.warning("Geen activiteiten gevonden. Probeer meer weken terug.")
    st.stop()

st.subheader("📋 Recente activiteiten")
st.dataframe(
    activities[["start_date_local","name","sport_type","distance_km","pace","average_heartrate","total_elevation_gain"]],
    use_container_width=True,
)

# =========[ Samenvatting + metrics ]=========
def summarize_block(df: pd.DataFrame, weeks: int = 4) -> dict:
    cutoff = datetime.now(timezone.utc) - timedelta(weeks=weeks)
    d = df[(df["start_date_local"] >= cutoff) & (df["sport_type"] == "Run")]
    if d.empty:
        return {"weeks": weeks, "km": 0, "longest_km": 0, "runs": 0, "avg_pace": None, "avg_hr": None, "elev": 0}
    km = float(d["distance_km"].sum())
    longest = float(d["distance_km"].max())
    runs = int(len(d))
    avg_pace = float(d["pace_min_per_km"].dropna().mean()) if d["pace_min_per_km"].notna().any() else None
    avg_hr = float(d["average_heartrate"].dropna().mean()) if d["average_heartrate"].notna().any() else None
    elev = float(d["total_elevation_gain"].dropna().sum())
    return {"weeks": weeks, "km": round(km,1), "longest_km": round(longest,1), "runs": runs,
            "avg_pace": avg_pace, "avg_hr": round(avg_hr,1) if avg_hr else None, "elev": round(elev,0)}

last4 = summarize_block(activities, 4)
last12 = summarize_block(activities, 12)

weeks_to_go = (
    datetime.combine(marathon_date, datetime.min.time(), tzinfo=timezone.utc) - datetime.now(timezone.utc)
).days // 7
today_str = datetime.now(timezone.utc).date().isoformat()
marathon_str = marathon_date.isoformat()
st.info(f"📅 Vandaag: {today_str} · Marathondatum: {marathon_str} · Weken tot marathon: {weeks_to_go}")

mcol1, mcol2, mcol3 = st.columns(3)
mcol1.metric("Km (laatste 4w)", last4["km"])
mcol2.metric("Langste run (km, 4w)", last4["longest_km"])
mcol3.metric("Weken tot marathon", max(weeks_to_go, 0))

def mins_to_pace_str(x):
    if x is None or not math.isfinite(x) or x <= 0: return "onbekend"
    m = int(x); s = int(round((x - m) * 60)); return f"{m}:{s:02d} /km"

# =========[ DIEPTE-ANALYSE: details/streams/zones ]=========
deep_insights_text = ""
if enable_deep:
    st.subheader("🔬 Diepte-analyse")
    zones = fetch_zones(access_token)
    runs_df = activities[activities["sport_type"] == "Run"].copy()
    if runs_df.empty:
        st.info("Geen hardloopactiviteiten in de gekozen periode.")
    else:
        # pak laatste N runs
        runs_df = runs_df.sort_values("start_date_local", ascending=False).head(deep_n)
        deep_rows = []
        for _, row in runs_df.iterrows():
            aid = int(row["id"]) if pd.notna(row["id"]) else None
            if not aid: continue

            # Details
            try:
                detail = fetch_activity_detail(access_token, aid)
            except requests.HTTPError:
                detail = {}

            # Streams
            streams = fetch_streams(access_token, aid)
            vel = streams.get("velocity_smooth", {}).get("data")
            hr = streams.get("heartrate", {}).get("data")
            cad = streams.get("cadence", {}).get("data")

            # HR-drift
            drift = hr_decoupling(vel, hr)

            # Time-in-zone
            tiz = compute_time_in_zones(hr, zones) if zones else None
            ez_qz = z_ratio_low_vs_high(tiz) if tiz else None
            easy_ratio = ez_qz[0] if ez_qz else None
            quality_ratio = ez_qz[1] if ez_qz else None

            deep_rows.append({
                "id": aid,
                "date": row["start_date_local"],
                "name": row["name"],
                "km": round(float(row["distance_km"]), 2),
                "avg_hr": row.get("average_heartrate", None),
                "avg_pace": mins_to_pace_str(row.get("pace_min_per_km")),
                "hr_drift": None if drift is None else round(float(drift)*100, 1),  # %
                "easy_ratio": None if easy_ratio is None else round(float(easy_ratio)*100,1),  # %
                "quality_ratio": None if quality_ratio is None else round(float(quality_ratio)*100,1),  # %
                "cad_mean": None if not cad else round(float(_safe_mean(cad)),1),
            })

            # kleine pauze tussen calls
            time.sleep(0.15)

        if deep_rows:
            deep_df = pd.DataFrame(deep_rows).sort_values("date", ascending=False)
            st.dataframe(deep_df, use_container_width=True)

            # Samenvatting voor prompt
            mean_drift = _safe_mean([r["hr_drift"] for r in deep_rows])
            mean_easy = _safe_mean([r["easy_ratio"] for r in deep_rows])
            mean_quality = _safe_mean([r["quality_ratio"] for r in deep_rows])
            mean_cad = _safe_mean([r["cad_mean"] for r in deep_rows])
            deep_insights = {
                "hr_drift_pct_avg": None if mean_drift is None else round(mean_drift,1),
                "time_in_zone_easy_pct_avg": None if mean_easy is None else round(mean_easy,1),
                "time_in_zone_quality_pct_avg": None if mean_quality is None else round(mean_quality,1),
                "cadence_mean_recent": None if mean_cad is None else round(mean_cad,1),
            }
            deep_insights_text = json.dumps(deep_insights, ensure_ascii=False)
        else:
            st.info("Geen geschikte runs gevonden voor diepte-analyse.")
else:
    st.caption("Tip: zet ‘🔬 Diepte-analyse’ aan om HR-drift, time-in-zone en cadans mee te nemen in het advies.")

# =========[ Profiel → dict ]=========
personal_profile = {
    "pt_days": pt_days,
    "pt_times": pt_times,
    "hockey_thursday": hockey,
    "hockey_time": hockey_time,
    "preferred_long_run_day": long_run_day,
    "rest_days": rest_days,
    "availability": weekly_availability,
    "extra_constraints": other_constraints,
}

# =========[ Advies genereren (knop) ]=========
st.markdown("---")
generate = st.button("🧠 Advies genereren")

if generate:
    st.session_state.background_text = updated_background or ""

    summary = {
        "periode_weken": weeks_back,
        "weken_tot_marathon": weeks_to_go,
        "doel": goal,
        "streeftijd": target_time if goal == "Tijdsspecifiek" else None,
        "km_laatste_4w": last4["km"],
        "langste_run_4w_km": last4["longest_km"],
        "gem_pace_4w": mins_to_pace_str(last4["avg_pace"]) if last4["avg_pace"] else None,
        "gem_hr_4w": last4["avg_hr"],
        "km_laatste_12w": last12["km"],
    }

    system_msg = (
        "Je bent een hardloopcoach. Produceer altijd veilige, concrete plannen.\n"
        "Regels:\n"
        "• Gebruik de meegegeven 'weken_tot_marathon' en absolute data als waarheid.\n"
        "• Geef NU alleen een gedetailleerd 4–6 weken mesocycle-plan.\n"
        "• Geef daarnaast een globale roadmap tot aan de marathon (base→build→peak→taper) op maand-/fase-niveau.\n"
        "• Taper pas in de laatste 2–3 weken vóór de marathon. Als weken_tot_marathon > 10: GEEN taper in het 4–6 weken plan.\n"
        "• Respecteer vaste afspraken (PT, hockey), voorkeurs-langeloopdag en rustdagen.\n"
        "• Vermijd zware sessies vlak vóór/na PT of andere zware sporten. Benoem blessurepreventie en alternatieven.\n"
    )

    user_msg = (
        "Context:\n"
        f"- Vandaag: {today_str}\n"
        f"- Marathondatum: {marathon_str}\n"
        f"- Weken tot marathon: {weeks_to_go}\n\n"
        "Opdracht:\n"
        "1) Maak een gedetailleerd schema voor de komende 4–6 weken (mesocycle) op basis van mijn recente Strava-data en mijn profiel.\n"
        "2) Maak daarnaast een globale planning tot aan de marathon met fases (base→build→peak→taper) en belangrijke mijlpalen (langste duurlopen), "
        "maar plan taper uitsluitend in de laatste 2–3 weken vóór de marathondatum.\n"
        "3) Respecteer PT/hockey/rustdagen en plan geen zware sessies vlak voor/na PT.\n\n"
        f"Samenvatting JSON: {json.dumps(summary, ensure_ascii=False)}\n"
        f"Persoonlijk profiel JSON: {json.dumps(personal_profile, ensure_ascii=False)}\n"
        f"Achtergrondinformatie:\n{st.session_state.background_text}\n"
    )

    if deep_insights_text:
        user_msg += f"\nDiepte-analyse samenvatting (recent): {deep_insights_text}\n"

    with st.spinner("Coachadvies genereren met ChatGPT…"):
        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.5,
        )
        advice = completion.choices[0].message.content

    st.subheader("🧠 Persoonlijk advies (ChatGPT)")
    st.markdown(advice)
    st.caption("Let op: dit is geen medisch advies.")
else:
    st.info("Bewerk eventueel de achtergrondtekst hierboven, kies of je ‘🔬 Diepte-analyse’ wilt, en klik daarna op **‘Advies genereren’**.")

