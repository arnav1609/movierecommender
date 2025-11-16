# app.py — robust CSS injection using components.html (final)
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
from recommender import MovieRecommender

CSV_PATH = "imdb_top_1000.csv"

st.set_page_config(page_title="Fun Movie Night", layout="wide", initial_sidebar_state="expanded")

# ----------------------
# Inject CSS via components.html (robust — prevents CSS from being printed as text)
# ----------------------
CSS_HTML = r"""
<link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap" rel="stylesheet">
<style>
:root{
  --accent1: #7b2ff7;
  --accent2: #ff6fd8;
  --accent3: #00d4ff;
  --glass: rgba(255,255,255,0.06);
  --card-bg: rgba(255,255,255,0.04);
  --muted: rgba(224,212,255,0.75);
}
html, body, [class*="css"]  { font-family: 'Roboto', sans-serif; }

/* Full page gradient background */
.stApp {
  background: radial-gradient(1200px 600px at 10% 10%, rgba(123,47,247,0.12), transparent 8%),
              radial-gradient(900px 500px at 90% 80%, rgba(255,111,216,0.10), transparent 6%),
              linear-gradient(180deg, #0c0b12 0%, #08101b 100%);
  min-height: 100vh;
  color: var(--muted);
}

/* floating soft blobs */
.bg-blob { position: fixed; z-index: 0; filter: blur(80px); opacity: 0.7; pointer-events: none; }
.blob1 { width: 420px; height: 420px; left: -80px; top: -60px;
  background: radial-gradient(circle at 30% 30%, rgba(123,47,247,0.5), rgba(123,47,247,0.15) 40%, transparent 60%);
  animation: float1 18s ease-in-out infinite;
}
.blob2 { width: 360px; height: 360px; right: -60px; bottom: -40px;
  background: radial-gradient(circle at 70% 70%, rgba(0,212,255,0.45), rgba(0,212,255,0.12) 45%, transparent 70%);
  animation: float2 20s ease-in-out infinite;
}
@keyframes float1 { 0% { transform: translateY(0) translateX(0) scale(1);} 50% { transform: translateY(20px) translateX(12px) scale(1.05);} 100% { transform: translateY(0) translateX(0) scale(1);} }
@keyframes float2 { 0% { transform: translateY(0) translateX(0) scale(1);} 50% { transform: translateY(-18px) translateX(-10px) scale(1.03);} 100% { transform: translateY(0) translateX(0) scale(1);} }

/* Header hero (glass card) */
.hero { position: relative; z-index: 2;
  background: linear-gradient(135deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
  border-radius: 16px; padding: 20px 28px;
  box-shadow: 0 10px 30px rgba(2,6,23,0.6), inset 0 1px 0 rgba(255,255,255,0.02);
  border: 1px solid rgba(255,255,255,0.03); margin-bottom: 18px;
}
.hero h1 { margin: 0; font-size: 36px; color: #fff; letter-spacing: -0.5px; }
.hero p { margin: 6px 0 0; color: rgba(255,255,255,0.75); }

/* Sidebar look (container) */
.stSidebar .css-1d391kg { background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.015)); border-radius: 12px; padding: 14px; }

/* Poster card */
.poster-card { background: var(--card-bg); border-radius: 12px; padding: 10px; transition: transform 0.18s ease, box-shadow 0.18s ease; border: 1px solid rgba(255,255,255,0.025); }
.poster-card:hover { transform: translateY(-6px); box-shadow: 0 14px 40px rgba(10,12,30,0.6); }
.poster-card .title { font-weight: 700; color: #fff; margin-top: 8px; margin-bottom: 4px; }
.poster-card .meta { color: rgba(224,212,255,0.75); font-size: 13px; }

/* Buttons */
.stButton>button { background: linear-gradient(90deg, var(--accent1), var(--accent2)); color: white; border-radius: 10px; padding: 8px 12px; border: none; box-shadow: 0 6px 18px rgba(123,47,247,0.12); }
.stButton>button:hover { transform: translateY(-2px); }

/* small tweaks for image look */
img { border-radius: 10px; object-fit: cover; }

/* ensure components overlay above blobs */
.streamlit-expander, .stApp, .css-1d391kg { z-index: 2; }

@media (max-width: 900px) { .hero h1 { font-size: 28px; } }
</style>
"""

components.html(CSS_HTML, height=0)  # inject CSS

# decorative blobs
components.html(
    """
    <div class="bg-blob blob1" aria-hidden="true"></div>
    <div class="bg-blob blob2" aria-hidden="true"></div>
    """,
    height=0,
)

# ----------------------
# Load recommender safely
# ----------------------
CSV_PATH_OBJ = Path(CSV_PATH)
if not CSV_PATH_OBJ.exists():
    st.error(f"CSV not found at path: {CSV_PATH}. Place your dataset there and reload.")
    st.stop()

@st.cache_resource(show_spinner=False)
def load_model(path: str):
    return MovieRecommender(path)

with st.spinner("Loading recommender model..."):
    try:
        recommender = load_model(CSV_PATH)
    except Exception as e:
        st.error("Failed to load recommender:")
        st.exception(e)
        st.stop()

# set session defaults
if 'last_recs' not in st.session_state:
    st.session_state['last_recs'] = None
if 'last_mode' not in st.session_state:
    st.session_state['last_mode'] = None
if 'last_snack' not in st.session_state:
    st.session_state['last_snack'] = None
if 'show_watchlist' not in st.session_state:
    st.session_state['show_watchlist'] = False

# -------- UI Header --------
components.html(
    """
    <div class="hero">
      <h1>🍿 Movie Night — Pick a mood, get a movie!</h1>
      <p>Friendly, fun picks — trailers, watchlist, and snack pairings. No algorithm labels, just great movie vibes.</p>
    </div>
    """,
    height=120,
)

# -------- Sidebar controls --------
with st.sidebar:
    st.markdown("## 🎛️ Controls")
    mood = st.selectbox("Pick your mood:", ["Feel-Good", "Mind-Bending", "Horror Night",
                                           "Romantic Vibes", "Action & Adrenaline",
                                           "Smart & Thought-Provoking", "Heartbreaking Drama"])
    actor = st.text_input("Favorite actor (optional):", placeholder="e.g. Tom Hanks")
    companion = st.selectbox("Tonight with:", ["Alone", "Friends", "Partner"])
    st.markdown("---")
    st.write("🎯 Quick actions")
    if st.button("🎲 Surprise Me!"):
        surprise = recommender.surprise_me(top_n=6)
        st.session_state['last_mode'] = 'surprise'
        st.session_state['last_recs'] = surprise
        st.session_state['last_snack'] = None
    if st.button("💾 View Watchlist"):
        st.session_state['show_watchlist'] = True

# -------- Main layout --------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Find movies")
    query = st.text_input("Search by movie title or describe what you'd like (e.g., 'time travel heist'):")
    row = st.columns([2, 1, 1])

    if row[0].button("🔎 Find"):
        if actor:
            actor_res = recommender.recommend_by_actor(actor, top_n=12)
            if actor_res:
                st.session_state['last_mode'] = 'actor'
                st.session_state['last_recs'] = actor_res
                st.session_state['last_snack'] = None
            else:
                res = recommender.recommend(query or actor, top_n=8)
                st.session_state['last_mode'] = 'query'
                st.session_state['last_recs'] = res
                st.session_state['last_snack'] = None
        elif query:
            res = recommender.recommend(query, top_n=8)
            st.session_state['last_mode'] = 'query'
            st.session_state['last_recs'] = res
            st.session_state['last_snack'] = None
        else:
            mood_block = recommender.recommend_by_mood(mood, top_n=8)
            st.session_state['last_mode'] = 'mood'
            st.session_state['last_recs'] = mood_block['recommendations']
            st.session_state['last_snack'] = mood_block['snack']

    if st.button(f"✨ Mood: {mood}"):
        mood_block = recommender.recommend_by_mood(mood, top_n=8)
        st.session_state['last_mode'] = 'mood'
        st.session_state['last_recs'] = mood_block['recommendations']
        st.session_state['last_snack'] = mood_block['snack']

    st.write("")
    recs = st.session_state.get('last_recs', None)

    if recs:
        st.markdown("### Recommendations")
        cols = st.columns(3)
        for i, r in enumerate(recs):
            col = cols[i % 3]
            with col:
                poster_url = r.get('poster') or "https://via.placeholder.com/300x450?text=No+Image"
                components.html(
                    f"""
                    <div class="poster-card" style="padding:6px;">
                        <img src="{poster_url}" alt="poster" width="100%" style="border-radius:10px;"/>
                        <div style="padding-top:8px;">
                          <div class="title">{r.get('title')}</div>
                          <div class="meta">{r.get('year','')} • {r.get('genre','')}</div>
                          <div class="meta">{"⭐ " + str(r.get('rating')) if r.get('rating') else ""}</div>
                        </div>
                    </div>
                    """,
                    height=380,
                )

                btn_row = st.columns([2, 1])
                if btn_row[0].button("▶ Trailer", key=f"tr_{i}"):
                    url = recommender.trailer_search_url(r.get('title'))
                    components.html(f"<script>window.open('{url}', '_blank');</script>", height=0)
                if btn_row[1].button("➕ Watchlist", key=f"wl_{i}"):
                    added = recommender.add_to_watchlist(r.get('title'))
                    if added:
                        st.success("Added to watchlist")
                    else:
                        st.info("Already in watchlist or not found")

        snack = st.session_state.get('last_snack', None)
        if snack:
            st.info(f"Snack pairing: **{snack}**")
    else:
        st.info("No recommendations yet — try Search, pick a Mood, or press Surprise Me!")

with col2:
    st.markdown("## 🎯 Quick Picks")
    if st.session_state.get('last_mode') == 'surprise':
        st.markdown("**Surprise picks for you**")

    wl = recommender.load_watchlist()
    if wl and len(wl) > 0:
        st.markdown("### Watchlist")
        for item in wl:
            poster = item.get('poster') or "https://via.placeholder.com/100x150?text=No+Image"
            st.image(poster, width=80)
            st.markdown(f"**{item.get('title')}** ({item.get('year')})")
        if st.button("Clear Watchlist"):
            recommender.save_watchlist([])
            st.experimental_rerun()
    else:
        st.write("Watchlist is empty — add movies you like!")

    st.markdown("---")
    st.markdown("### Tips")
    st.write("- Use 'Surprise Me' if you can't decide.")
    st.write("- Click Trailer to watch the trailer on YouTube.")
    st.write("- Add your favorite actor to find more movies with them.")

# Footer
st.markdown("---")
st.caption("Built with ❤️ — enjoy your movie night!")
