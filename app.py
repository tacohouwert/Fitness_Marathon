# app.py
import os
import json
import math
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
import streamlit as st
from dateutil import tz
from openai import OpenAI

# =========[ Streamlit basis ]=========
st.set_page_config(
    page_title="NY Marathon Coach",
    page_icon="🏃‍♂️",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.title("🏃‍♂️ NY Marathon Coach — Strava → Persoonlijk advies")

# =========[ Config uit Secrets ]=========
STRAVA_CLIENT_ID = st.secrets["STRAVA_CLIENT_ID"]
STRAVA_CLIENT_SECRET = st.secrets["STRAVA_CLIENT_SECRET"]
STRAVA_REFRESH_TOKEN = st.secrets["STRAVA_REFRESH_TOKEN"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
OPENAI_MODEL = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")

STRAVA_TOKEN_URL = "https://www.strava.com/oauth/token"
STRAVA_API_BASE = "https://www.strava.com/api/v3"

# OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# =========[ Sidebar: Persoonlijk profiel ]=========
with st.sidebar:
    st.header("👤 Persoonlijk profiel")
    pt_days = st.multiselect(
        "PT-dagen",
        ["maandag", "dinsdag", "woensdag", "donderdag", "vrijdag"],
        default=["maandag", "woensdag"],
    )
    pt_times = st.text_input(
        "PT tijden (bijv. ma 07:30, wo 18:00)",
        value="ma 07:30, wo 18:00",
    )

    hockey = st.checkbox("Ik hockey op donderdag", value=True)
    hockey_time = st.text_input("Hockey tijd", value="do 20:30–22:00")

    long_run_day = st.selectbox(
        "Voorkeursdag lange duurloop", ["zaterdag", "zondag", "vrijdag"], index=0
    )
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
        value="Kleine gevoeligheid achilles links; liever geen intensieve intervallen op PT-dagen.",
        placeholder="Blessurehistorie, voorkeuren, reistijd, etc.",
    )
    st.caption("Dit profiel wordt meegenomen in het coach-advies.")

# =========[ Helpers ]=========
def bearer_header(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}

@st.cache_data(ttl=50 * 60)  # 50 min cache; zet uit als je actief debugt
def get_access_token(refresh_token: str) -> dict:
    """Vraag met refresh_token een verse access_token op."""
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
    return resp.json()  # bevat access_token, expires_at, athlete, etc.

def _fmt_pace(min_per_km: float | None) -> str | None:
    if min_per_km is None or not math.isfinite(min_per_km) or min_per_km <= 0:
        return None
    m = int(min_per_km)
    s = int(round((min_per_km - m) * 60))
    return f"{m}:{s:02d} /km"

def _rate_limit_headers(resp: requests.Response):
    return (
        resp.headers.get("X-RateLimit-Usage"),
        resp.headers.get("X-RateLimit-Limit"),
        resp.headers.get("X-RateLimit-Reset"),
    )

def _fetch_activities_nocache(
    access_token: str, after_ts: int, per_page: int = 50, max_pages: int = 3
) -> pd.DataFrame:
    """Ruwe fetch zonder cache; met eenvoudige backoff en nette fouten."""
    all_rows = []
    for page in range(1, max_pages + 1):
        params = {"per_page": per_page, "page": page, "after": after_ts}
        r = requests.get(
            f"{STRAVA_API_BASE}/athlete/activities",
            headers=bearer_header(access_token),
            params=params,
            timeout=30,
        )

        if r.status_code == 429:
            usage, limit, reset = _rate_limit_headers(r)
            st.error(
                f"Strava rate limit bereikt (429). Usage/Limit: {usage}/{limit}. "
                "Probeer het over enkele minuten opnieuw."
            )
            r.raise_for_status()

        if r.status_code == 401:
            st.error(
                "Strava 401 (Unauthorized): access token ongeldig. "
                "Controleer je STRAVA_REFRESH_TOKEN en maak zo nodig een nieuw token met scopes "
                "`read,activity:read_all`."
            )
            r.raise_for_status()

        r.raise_for_status()
        chunk = r.json()
        if not chunk:
            break
        all_rows.extend(chunk)
        if len(chunk) < per_page:
            break
        time.sleep(0.4)  # kleine pauze tussen pagina's

    if not all_rows:
        return pd.DataFrame()

    df = pd.json_normalize(all_rows)

    # nuttige kolommen selecteren
    cols = [
        "name",
        "type",
        "sport_type",
        "start_date_local",
        "distance",
        "moving_time",
        "elapsed_time",
        "total_elevation_gain",
        "average_speed",
        "max_speed",
        "average_heartrate",
        "max_heartrate",
        "has_heartrate",
        "suffer_score",
        "private",
        "visibility",
        "kudos_count",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df[cols].copy()

    # conversies/afgeleiden
    df["start_date_local"] = pd.to_datetime(df["start_date_local"])
    df["distance_km"] = df["distance"].fillna(0) / 1000.0
    df["pace_min_per_km"] = df["moving_time"] / (df["distance_km"] * 60)
    df["pace"] = df["pace_min_per_km"].apply(_fmt_pace)

    return df.sort_values("start_date_local", ascending=False)

@st.cache_data(ttl=15 * 60)  # 15 min cache om 429 te vermijden
def fetch_activities_cached(access_token: str, after_ts: int, per_page: int, max_pages: int) -> pd.DataFrame:
    return _fetch_activities_nocache(access_token, after_ts, per_page=per_page, max_pages=max_pages)

def summarize_block(df: pd.DataFrame, weeks: int = 4) -> dict:
    cutoff = datetime.now(tz=timezone.utc) - timedelta(weeks=weeks)
    d = df[df["start_date_local"] >= cutoff]
    if d.empty:
        return {
            "weeks": weeks,
            "km": 0,
            "longest_km": 0,
            "runs": 0,
            "avg_pace": None,
            "avg_hr": None,
            "elev": 0,
        }
    km = float(d["distance_km"].sum())
    longest = float(d["distance_km"].max())
    runs = int((d["sport_type"] == "Run").sum()) if "sport_type" in d.columns else len(d)
    avg_pace_series = d["pace_min_per_km"].replace([pd.NA, pd.NaT], float("nan")).dropna()
    avg_pace = float(avg_pace_series.mean()) if len(avg_pace_series) else None
    avg_hr = d["average_heartrate"].dropna().mean() if "average_heartrate" in d.columns else None
    elev = float(d["total_elevation_gain"].dropna().sum())
    return {
        "weeks": weeks,
        "km": round(km, 1),
        "longest_km": round(longest, 1),
        "runs": runs,
        "avg_pace": avg_pace,
        "avg_hr": round(avg_hr, 1) if avg_hr else None,
        "elev": round(elev, 0),
    }

def mins_to_pace_str(x):
    if x is None or not math.isfinite(x) or x <= 0:
        return "onbekend"
    m = int(x)
    s = int(round((x - m) * 60))
    return f"{m}:{s:02d} /km"

# =========[ UI: parameters ]=========
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    weeks_back = st.slider("Hoeveel weken terug ophalen?", 4, 26, 8)  # lager default helpt tegen 429
with col2:
    marathon_date = st.date_input(
        "Marathon datum",
        value=datetime(datetime.now().year + 1, 11, 1).date(),
        help="Gebruik de exacte datum van de NYC Marathon die jij gaat lopen.",
    )
with col3:
    goal = st.selectbox("Doel", ["Uitlopen", "PR lopen", "Tijdsspecifiek"], index=0)
    target_time = st.text_input("Streeftijd (hh:mm:ss)", value="", help="Alleen invullen bij Tijdsspecifiek.")

# =========[ Background.txt → eerst bewerken, dan pas genereren ]=========
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

st.caption("Wijzigingen worden niet automatisch naar het bestand geschreven op Streamlit Cloud; ze gelden voor deze sessie.")

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
    st.warning("Geen activiteiten gevonden in de gekozen periode. Probeer meer weken of controleer je Strava-data.")
    st.stop()

st.subheader("📋 Recente activiteiten")
st.dataframe(
    activities[
        [
            "start_date_local",
            "name",
            "sport_type",
            "distance_km",
            "pace",
            "average_heartrate",
            "total_elevation_gain",
        ]
    ],
    use_container_width=True,
)

# =========[ Samenvatting + metrics ]=========
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
generate = st.button("🧠 Advies genereren op basis van bovenstaande achtergrond + Strava")

if generate:
    # Update session background met de bewerkte tekst
    st.session_state.background_text = updated_background or ""

    # =========[ ChatGPT-advies ]=========
    summary = {
        "periode_weken": weeks_back,
        "weken_tot_marathon": weeks_to_go,
        "doel": goal,
        "streeftijd": target_time if goal == "Tijdsspecifiek" else None,
        "km_laatste_4w": last4["km"],
        "langste_run_4w_km": last4["longest_km"],
        "gem_pace_4w": mins_to_pace_str(last4["avg_pace"]),
        "gem_hr_4w": last4["avg_hr"],
        "km_laatste_12w": last12["km"],
    }

    # System prompt met harde regels (geen te vroege taper)
    system_msg = (
        "Je bent een hardloopcoach. Produceer altijd veilige, concrete plannen.\n"
        "Regels:\n"
        "• Gebruik de meegegeven 'weken_tot_marathon' en absolute data als waarheid.\n"
        "• Geef NU alleen een gedetailleerd 4–6 weken mesocycle-plan.\n"
        "• Geef daarnaast een globale roadmap tot aan de marathon (base→build→peak→taper) op maand-/fase-niveau.\n"
        "• Taper pas in de laatste 2–3 weken vóór de marathon. Als weken_tot_marathon > 10: GEEN taper in het 4–6 weken plan.\n"
        "• Respecteer vaste afspraken (PT, hockey), voorkeurs-langeloopdag en rustdagen.\n"
        "• Vermijd zware sessies vlak vóór/na PT of zware sporten. Benoem blessurepreventie en alternatieven.\n"
    )

    # User prompt met geüpdatete achtergrondtekst
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
        f"Achtergrondinformatie (handmatig bijgewerkt):\n{st.session_state.background_text}\n"
    )

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
    st.caption(
        "Let op: dit is geen medisch advies. Luister naar je lichaam en raadpleeg een professional bij klachten."
    )
else:
    st.info("Bewerk eventueel de achtergrondtekst hierboven en klik daarna op **‘Advies genereren’**.")
