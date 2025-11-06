# app.py
import os, json, math, time
from datetime import datetime, timedelta, timezone
import numpy as np, pandas as pd, requests, streamlit as st
from openai import OpenAI
import matplotlib.pyplot as plt

# =========[ Streamlit setup ]=========
st.set_page_config(page_title="NY Marathon Coach", page_icon="🏃‍♂️", layout="wide")
st.title("🏃‍♂️ NY Marathon Coach — Strava → Persoonlijk advies")

# =========[ Secrets controle ]=========
REQUIRED = ["STRAVA_CLIENT_ID","STRAVA_CLIENT_SECRET","STRAVA_REFRESH_TOKEN","OPENAI_API_KEY"]
missing = [k for k in REQUIRED if k not in st.secrets]
if missing:
    st.error(f"❌ Ontbrekende secrets: {', '.join(missing)}\nVoeg ze toe in Streamlit Cloud → Settings → Secrets.")
    st.stop()

STRAVA_CLIENT_ID = st.secrets["STRAVA_CLIENT_ID"]
STRAVA_CLIENT_SECRET = st.secrets["STRAVA_CLIENT_SECRET"]
STRAVA_REFRESH_TOKEN = st.secrets["STRAVA_REFRESH_TOKEN"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
MODEL = st.secrets.get("OPENAI_MODEL","gpt-4o-mini")
client = OpenAI(api_key=OPENAI_API_KEY)

STRAVA_API = "https://www.strava.com/api/v3"

def bearer(t): return {"Authorization": f"Bearer {t}"}

# =========[ Tokens en Strava API ]=========
@st.cache_data(ttl=50*60)
def get_access_token(refresh_token):
    r = requests.post("https://www.strava.com/oauth/token", data={
        "client_id": STRAVA_CLIENT_ID,
        "client_secret": STRAVA_CLIENT_SECRET,
        "grant_type": "refresh_token",
        "refresh_token": refresh_token})
    r.raise_for_status(); return r.json()

@st.cache_data(ttl=15*60)
def fetch_activities(token, after_ts, pages=3):
    all_rows=[]
    for p in range(1,pages+1):
        r=requests.get(f"{STRAVA_API}/athlete/activities",headers=bearer(token),
                       params={"per_page":50,"page":p,"after":after_ts},timeout=30)
        if not r.ok: break
        chunk=r.json()
        if not chunk: break
        all_rows+=chunk
        if len(chunk)<50: break
        time.sleep(0.3)
    if not all_rows: return pd.DataFrame()
    df=pd.json_normalize(all_rows)
    
    df["start_date_local"] = pd.to_datetime(df["start_date_local"], utc=True, errors="coerce")


    df["distance_km"]=df["distance"]/1000
    df["pace_min_per_km"]=df["moving_time"]/(df["distance_km"]*60)
    df["pace"]=df["pace_min_per_km"].apply(lambda x:f"{int(x)}:{int((x-int(x))*60):02d}/km" if x and x>0 else None)
    return df.sort_values("start_date_local",ascending=False)

@st.cache_data(ttl=60*60)
def fetch_streams(token, aid):
    try:
        r=requests.get(f"{STRAVA_API}/activities/{aid}/streams",headers=bearer(token),
                       params={"keys":"time,distance,velocity_smooth,heartrate,cadence,altitude","key_by_type":"true"},timeout=30)
        if r.ok: return r.json()
    except: pass
    return {}

@st.cache_data(ttl=24*60*60)
def fetch_zones(token):
    r=requests.get(f"{STRAVA_API}/athlete/zones",headers=bearer(token))
    return r.json() if r.ok else {}

# =========[ Analyse helpers ]=========
def hr_decoupling(vel,hr):
    if not vel or not hr: return None
    n=min(len(vel),len(hr))
    if n<240: return None
    v=np.array(vel[:n]); h=np.array(hr[:n])
    half=n//2
    r1=v[:half].mean()/max(h[:half].mean(),1)
    r2=v[half:].mean()/max(h[half:].mean(),1)
    return (r2-r1)/r1*100

def time_in_zones(hr, zones):
    zdef=zones.get("heart_rate",{}).get("zones",[])
    if not hr or not zdef: return {}
    counts=[0]*len(zdef)
    for h in hr:
        for i,z in enumerate(zdef):
            if z.get("min",0)<=h<=z.get("max",1e9):
                counts[i]+=1; break
    tot=sum(counts) or 1
    return {f"Z{i+1}":round(c/tot*100,1) for i,c in enumerate(counts)}

# =========[ Sidebar profiel — originele opties terug ]=========
with st.sidebar:
    st.header("👤 Persoonlijk profiel")
    pt_days = st.multiselect("PT-dagen",["maandag","dinsdag","woensdag","donderdag","vrijdag"],["maandag","woensdag"])
    pt_times = st.text_input("PT tijden", "ma 20:00, wo 18:45")
    hockey = st.checkbox("Ik hockey op donderdag", True)
    hockey_time = st.text_input("Hockey tijd", "do 20:00–21:30")
    long_run_day = st.selectbox("Voorkeursdag lange duurloop",["zaterdag","zondag","vrijdag"],0)
    rest_days = st.multiselect("Vaste rustdagen",["maandag","dinsdag","woensdag","donderdag","vrijdag","zaterdag","zondag"],[])
    st.markdown("—")
    weekly_availability = st.text_area("Beschikbaarheid (vrije vensters)", "di 07:00–08:00, vr 12:00–13:00")
    other_constraints = st.text_area("Extra context / beperkingen", "Liever geen intensieve intervallen op PT-dagen.")
    st.caption("Dit profiel wordt meegenomen in het coach-advies.")
    st.divider()
    st.caption(f"Laatste update: {datetime.now().strftime('%H:%M:%S')}")

# =========[ Parameters bovenin ]=========
col1,col2,col3 = st.columns([1,1,2])
with col1: weeks_back = st.slider("Weken terug ophalen",4,26,8)
with col2: marathon_date = st.date_input("Marathon datum", datetime(datetime.now().year+1,11,1).date())
with col3: goal = st.selectbox("Doel",["Uitlopen","PR lopen","Tijdsspecifiek"])
deep = st.checkbox("🔬 Diepte-analyse",value=True)

# =========[ Achtergrondinformatie ]=========
bg_path = os.path.join(os.getcwd(),"background.txt")
background = open(bg_path,"r",encoding="utf-8").read().strip() if os.path.exists(bg_path) else ""
background = st.text_area("📝 Achtergrondinformatie (optioneel)", value=background, height=180)

# =========[ Strava data ophalen ]=========
after = int((datetime.now(timezone.utc) - timedelta(weeks=weeks_back)).timestamp())
with st.spinner("📡 Strava-data ophalen..."):
    token = get_access_token(STRAVA_REFRESH_TOKEN)
    access = token["access_token"]
    acts = fetch_activities(access, after)
if acts.empty: st.warning("Geen activiteiten gevonden."); st.stop()

# =========[ Overzicht van recente Strava-activiteiten ]=========
if not acts.empty:
    st.subheader("📋 Strava-activiteiten")

    with st.expander("🏃 Toon recente activiteiten (klik om te openen)"):
        st.dataframe(
            acts[
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
        st.caption(
            "Hier zie je je recente activiteiten die uit Strava zijn opgehaald. "
            "De lijst bevat afstand (km), tempo, gemiddelde hartslag en hoogtemeters. "
            "Klik op de kolomkoppen om te sorteren."
        )
else:
    st.warning("Geen activiteiten gevonden in de geselecteerde periode.")


# =========[ Samenvatting (UTC fix) ]=========
def summary_block(df, w=4):
    now_utc = datetime.now(timezone.utc)
    cutoff = now_utc - timedelta(weeks=w)
    d = df[df["start_date_local"] >= cutoff]
    if d.empty: return {"weeks":w,"km":0,"longest":0,"runs":0,"pace":None,"hr":None}
    return {
        "weeks":w,
        "km":round(d["distance_km"].sum(),1),
        "longest":round(d["distance_km"].max(),1),
        "runs":len(d),
        "pace":round(d["pace_min_per_km"].mean(),2),
        "hr":round(d["average_heartrate"].mean(),1) if d["average_heartrate"].notna().any() else None
    }

s4,s12 = summary_block(acts,4), summary_block(acts,12)
st.metric("Km (4w)",s4["km"]); st.metric("Langste run (km)",s4["longest"])

# =========[ Diepte-analyse ]=========
deep_insights = {}
if deep:
    zones = fetch_zones(access)
    runs = acts[acts["sport_type"] == "Run"].head(8)
    drifts = []

    for _, r in runs.iterrows():
        sid = int(r["id"])
        streams = fetch_streams(access, sid)

        drift = hr_decoupling(
            streams.get("velocity_smooth", {}).get("data"),
            streams.get("heartrate", {}).get("data"),
        )

        tiz = time_in_zones(
            streams.get("heartrate", {}).get("data"), zones
        )

        drifts.append({
            "date": r["start_date_local"],
            "hr_drift": drift,
            **tiz
        })

    ddf = pd.DataFrame(drifts).dropna(subset=["hr_drift"])

    if not ddf.empty:
        st.subheader("🔬 Diepte-analyse")

        # Uitklapbare sectie voor de tabel
        with st.expander("📋 Toon tabel met HR-drift per run"):
            st.dataframe(
                ddf.style.background_gradient(subset=["hr_drift"], cmap="coolwarm")
            )
            st.caption(
                "Elke regel toont de hartslag-drift per run (%). "
                "Positieve waarden = oplopende hartslag → meer vermoeidheid; "
                "negatieve waarden = stabiel of beter herstel."
            )

        # Uitklapbare sectie voor de grafiek
        with st.expander("📉 Toon HR-drift-grafiek"):
            fig, ax = plt.subplots()
            ax.plot(ddf["date"], ddf["hr_drift"], "o-", label="HR-drift (%)")
            ax.axhline(5, color="g", ls="--", label="Goede zone (<5%)")
            ax.axhline(10, color="r", ls="--", label="Te hoog (>10%)")
            ax.set_ylabel("HR-drift (%)")
            ax.set_xlabel("Datum")
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)
            st.caption(
                "De grafiek toont de HR-drift (%) per run. "
                "Een lagere waarde wijst op betere aerobe efficiëntie."
            )

        # Gemiddelde drift berekenen
        deep_insights["gem_hr_drift"] = round(ddf["hr_drift"].mean(), 1)
    else:
        st.info("Geen HR-driftdata beschikbaar voor de geselecteerde runs.")

# =========[ Persoonlijk profiel opslaan voor ChatGPT ]=========
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
    # Bepaal context op basis van gekozen marathondatum
    today_utc = datetime.now(timezone.utc).date()
    weeks_to_go = (marathon_date - today_utc).days // 7
    today_str = today_utc.isoformat()
    marathon_str = marathon_date.isoformat()

    # Samenvatting o.b.v. je bestaande aggregaten (s4/s12)
    summary = {
        "periode_weken_terug_ingelezen": weeks_back,
        "weken_tot_marathon": weeks_to_go,
        "doel": goal,
        "km_laatste_4w": s4["km"],
        "langste_run_4w_km": s4["longest"],
        "gem_pace_4w_min_per_km": s4["pace"],
        "gem_hr_4w": s4["hr"],
        "km_laatste_12w": s12["km"],
    }

    # Strikte systeemprompt om verkeerde aannames te voorkomen
    system_msg = (
        "Je bent een ervaren marathoncoach.\n"
        "HARD-CONSTRAINTS (MOET je volgen):\n"
        f"1) De marathondatum is {marathon_str}. Er zijn precies {{weken_tot_marathon}} weken te gaan.\n"
        "2) Maak ALLEEN een concreet trainingsschema voor de komende 4 weken (week 1 t/m 4 vanaf vandaag). "
        "Gebruik geen 5e of 6e week in het concrete schema.\n"
        "3) Geef daarnaast een KORTE macro-roadmap (base→build→peak→taper) tot de marathondatum, "
        "zonder week-tot-week details. Plan TAPER pas als weken_tot_marathon ≤ 3.\n"
        "4) Respecteer vaste afspraken (PT, hockey), voorkeurs-langeloopdag en rustdagen. "
        "Vermijd zware sessies vlak voor/na PT en geef blessurepreventie mee.\n"
        "5) Noem expliciet: 'Er zijn X weken tot de marathon op YYYY-MM-DD', waarbij X de aangeleverde waarde is."
    )

    # Gebruikersbericht met expliciete waarde-injectie (we geven X letterlijk mee)
    user_payload = {
        "vandaag": today_str,
        "marathondatum": marathon_str,
        "weken_tot_marathon": weeks_to_go,          # <- expliciet en leidend
        "samenvatting": summary,
        "profiel": personal_profile,
        "diepte_analyse": deep_insights,            # bv. gem_hr_drift
        "achtergrond": background
    }
    user_msg = json.dumps(user_payload, ensure_ascii=False, indent=2)

    with st.spinner("Coachadvies genereren met ChatGPT…"):
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_msg.replace("{weken_tot_marathon}", str(weeks_to_go))},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.4,
        )
        advice = completion.choices[0].message.content

    st.subheader("🧠 Persoonlijk advies (ChatGPT)")
    st.markdown(advice)
    st.caption(
        "Let op: dit is geen medisch advies. Het schema is concreet voor 4 weken; de roadmap is globaal tot de marathondatum."
    )
else:
    st.info("Bewerk eventueel de achtergrondtekst hierboven en klik daarna op **‘Advies genereren’**.")















