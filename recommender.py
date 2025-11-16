# recommender.py
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

class MovieRecommender:
    """
    Friendly movie recommender:
      - content-based TF-IDF + TruncatedSVD
      - optional item-item CF (if ratings CSV provided)
      - mood mapping, actor-filter, surprise-me, trailer link helper, snack pairing, watchlist
    """
    def __init__(self, csv_path: str, ratings_csv: Optional[str] = None,
                 n_components: int = 50, max_features: int = 3000):
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"{csv_path} not found.")
        self.df = pd.read_csv(self.csv_path)

        # Normalize column names
        self._colmap = {c.lower(): c for c in self.df.columns}
        def get_col(*candidates):
            for c in candidates:
                if c.lower() in self._colmap:
                    return self._colmap[c.lower()]
            return None

        # Common columns expected in imdb_top_1000.csv
        self.COL_TITLE = get_col("Series_Title", "title", "name")
        self.COL_GENRE = get_col("Genre", "genres", "genre")
        self.COL_OVERVIEW = get_col("Overview", "overview", "description")
        self.COL_POSTER = get_col("Poster_Link", "poster", "poster_link")
        self.COL_RATING = get_col("IMDB_Rating", "rating", "imdb_rating")
        self.COL_YEAR = get_col("Released_Year", "Year", "year")
        self.COL_MOVIEID = get_col("movieId", "id", "movie_id")

        # Fill defaults if missing
        if not self.COL_GENRE:
            self.df['Genre'] = ""
            self.COL_GENRE = 'Genre'
        if not self.COL_OVERVIEW:
            self.df['Overview'] = ""
            self.COL_OVERVIEW = 'Overview'
        if not self.COL_POSTER:
            self.df['Poster_Link'] = ""
            self.COL_POSTER = 'Poster_Link'
        if not self.COL_RATING:
            self.df['IMDB_Rating'] = np.nan
            self.COL_RATING = 'IMDB_Rating'
        if not self.COL_TITLE:
            raise ValueError("No title-like column found in movies CSV.")
        if not self.COL_MOVIEID:
            self.df['movieId'] = range(1, len(self.df) + 1)
            self.COL_MOVIEID = 'movieId'
        if not self.COL_YEAR:
            # try extract year from title if present
            self.df['Released_Year'] = self.df[self.COL_TITLE].astype(str).str.extract(r'\((\d{4})\)')
            self.COL_YEAR = 'Released_Year'

        # Combined text for TF-IDF
        self.df['combined_features'] = (
            self.df[self.COL_GENRE].fillna('').astype(str) + " " +
            self.df[self.COL_OVERVIEW].fillna('').astype(str) + " " +
            self.df[self.COL_TITLE].fillna('').astype(str)
        )

        # Config
        self.n_components = n_components
        self.max_features = max_features

        # Models & matrices
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.svd_model: Optional[TruncatedSVD] = None
        self.item_vectors: Optional[np.ndarray] = None
        self.content_sim_matrix: Optional[np.ndarray] = None

        # CF placeholders
        self.ratings_csv = ratings_csv
        self.item_item_model: Optional[NearestNeighbors] = None
        self.item_id_index_map: Optional[pd.Index] = None
        self.item_cf_vectors: Optional[np.ndarray] = None

        # watchlist
        self.watchlist_file = Path("watchlist.json")

        # mood map (easy to extend)
        self.mood_map = {
            "Feel-Good": ["comedy", "family", "animation", "feel-good", "romance"],
            "Mind-Bending": ["sci-fi", "thriller", "mystery", "psychological"],
            "Horror Night": ["horror", "thriller", "supernatural"],
            "Romantic Vibes": ["romance", "drama", "comedy"],
            "Action & Adrenaline": ["action", "adventure", "crime"],
            "Smart & Thought-Provoking": ["drama", "biography", "history", "documentary"],
            "Heartbreaking Drama": ["drama", "romance"]
        }

        # snack pairs
        self.snack_map = {
            "Horror Night": "Popcorn + Cold Drink 🍿",
            "Feel-Good": "Nachos + Soda 🧀",
            "Romantic Vibes": "Chocolate + Tea 🍫",
            "Action & Adrenaline": "Chips + Energy Drink ⚡",
            "Mind-Bending": "Coffee + Dark Chocolate ☕",
            "Smart & Thought-Provoking": "Herbal Tea + Nuts 🌿",
            "Heartbreaking Drama": "Ice Cream + Tissues 🍨"
        }

        # Build content model now
        self.build()

    def build(self):
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=self.max_features)
        tfidf = self.tfidf_vectorizer.fit_transform(self.df['combined_features'].fillna(''))

        # TruncatedSVD on sparse matrix
        n_comp = min(self.n_components, tfidf.shape[1] - 1) if tfidf.shape[1] > 1 else 1
        self.svd_model = TruncatedSVD(n_components=max(1, n_comp), random_state=42)
        self.item_vectors = self.svd_model.fit_transform(tfidf)

        # similarity matrix
        self.content_sim_matrix = cosine_similarity(self.item_vectors)

        # optional CF
        if self.ratings_csv:
            self._build_item_item_cf()

    def _build_item_item_cf(self):
        rpath = Path(self.ratings_csv)
        if not rpath.exists():
            print(f"[WARN] ratings file {self.ratings_csv} not found. Skipping CF.")
            return
        ratings = pd.read_csv(rpath)
        possible_movieid_cols = [c for c in ['movieId','movie_id','id'] if c in ratings.columns]
        if possible_movieid_cols:
            mid_col = possible_movieid_cols[0]
            pivot = ratings.pivot_table(index='userId', columns=mid_col, values='rating').fillna(0)
        else:
            # try by title
            title_col = None
            for c in ['title','movieTitle','Series_Title']:
                if c in ratings.columns:
                    title_col = c
                    break
            if title_col:
                merged = ratings.merge(self.df[[self.COL_TITLE,self.COL_MOVIEID]].rename(columns={self.COL_TITLE:'title', self.COL_MOVIEID:'movieId'}),
                                       left_on=title_col, right_on='title', how='left')
                pivot = merged.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
            else:
                print("[WARN] Could not find movie identifier in ratings CSV. Skipping CF.")
                return

        self.item_id_index_map = pivot.columns
        item_matrix = pivot.T.values
        self.item_cf_vectors = item_matrix
        if item_matrix.shape[0] < 2:
            print("[WARN] Not enough items for CF.")
            return
        nn = NearestNeighbors(metric='cosine', algorithm='brute')
        nn.fit(item_matrix)
        self.item_item_model = nn

    # -----------------------------
    # Helpers
    # -----------------------------
    def _fuzzy_match_title(self, title: str, cutoff: float = 0.6) -> Tuple[Optional[str], float]:
        candidates = self.df[self.COL_TITLE].astype(str).tolist()
        if not title or not candidates:
            return None, 0.0
        matches = difflib.get_close_matches(title, candidates, n=1, cutoff=cutoff)
        if matches:
            best = matches[0]
            ratio = difflib.SequenceMatcher(None, title, best).ratio()
            return best, ratio
        return None, 0.0

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        if arr is None or len(arr) == 0:
            return np.array([])
        arr = np.array(arr, dtype=float)
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-9:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    # -----------------------------
    # Core recommend
    # -----------------------------
    def recommend(self, seed: str, top_n: int = 6, method: str = "hybrid", alpha: float = 0.7, fuzzy: bool = True) -> List[Dict]:
        """
        seed: either a movie title or a natural language query
        method: "content", "cf", "hybrid"
        """
        if not seed:
            return []

        # try exact title or fuzzy match
        matched_title = None
        if seed in self.df[self.COL_TITLE].values:
            matched_title = seed
        elif fuzzy:
            matched_title, ratio = self._fuzzy_match_title(seed)
            # if no fuzzy match, matched_title stays None -> treat seed as query

        idx = None
        if matched_title:
            idx = int(self.df[self.df[self.COL_TITLE] == matched_title].index[0])

        # content scores: either similarity row or query->items similarity
        if idx is not None:
            content_scores = self.content_sim_matrix[idx]
        else:
            q_vec = self.tfidf_vectorizer.transform([seed])
            q_red = self.svd_model.transform(q_vec)
            content_scores = cosine_similarity(q_red, self.item_vectors).flatten()

        # cf scores (if available)
        cf_scores = np.zeros(len(self.df))
        if method in ("cf", "hybrid") and self.item_item_model is not None and idx is not None:
            movie_id = self.df.iloc[idx][self.COL_MOVIEID]
            try:
                pos = list(self.item_id_index_map).index(movie_id)
                distances, indices = self.item_item_model.kneighbors([self.item_cf_vectors[pos]], n_neighbors=min(50, len(self.item_id_index_map)))
                neighbor_ids = [self.item_id_index_map[i] for i in indices.flatten()]
                neighbor_scores = 1 - distances.flatten()  # convert distance -> similarity-like
                cf_map = dict(zip(neighbor_ids, neighbor_scores))
                df_mids = self.df[self.COL_MOVIEID].tolist()
                cf_scores = np.array([cf_map.get(mid, 0.0) for mid in df_mids])
            except ValueError:
                cf_scores = np.zeros(len(self.df))

        # choose final
        if method == "content":
            final = content_scores
        elif method == "cf":
            final = cf_scores
        else:
            nc = self._normalize(content_scores)
            nf = self._normalize(cf_scores)
            final = alpha * nc + (1 - alpha) * nf

        # exclude the seed if matched
        if idx is not None:
            final[idx] = -1.0

        top_idx = np.argsort(final)[::-1][:top_n]
        results = []
        for i in top_idx:
            row = self.df.iloc[i]
            results.append({
                "title": str(row[self.COL_TITLE]),
                "movieId": int(row[self.COL_MOVIEID]) if pd.notna(row[self.COL_MOVIEID]) else None,
                "rating": float(row[self.COL_RATING]) if pd.notna(row[self.COL_RATING]) else None,
                "genre": str(row[self.COL_GENRE]) if pd.notna(row[self.COL_GENRE]) else "",
                "poster": str(row[self.COL_POSTER]) if pd.notna(row[self.COL_POSTER]) else "",
                "year": str(row[self.COL_YEAR]) if pd.notna(row[self.COL_YEAR]) else "",
                "score": float(final[i])
            })
        return results

    # -----------------------------
    # Fun helpers: mood, actor, surprise
    # -----------------------------
    def recommend_by_mood(self, mood: str, top_n: int = 6) -> Dict:
        keywords = self.mood_map.get(mood, [])
        query = " ".join(keywords) if keywords else mood
        recs = self.recommend(query, top_n=top_n, method="content", fuzzy=False)
        snack = self.snack_map.get(mood, "Popcorn 🍿")
        return {"mood": mood, "snack": snack, "recommendations": recs}

    def recommend_by_actor(self, actor_name: str, top_n: int = 12) -> List[Dict]:
        star_cols = [c for c in self.df.columns if any(s in c.lower() for s in ("star","actor","cast"))]
        if star_cols:
            mask = self.df[star_cols].fillna('').apply(lambda row: actor_name.lower() in " ".join(row.astype(str)).lower(), axis=1)
            filtered = self.df[mask]
            if filtered.empty:
                return []
            filtered = filtered.sort_values(by=self.COL_RATING if self.COL_RATING in self.df.columns else self.COL_TITLE, ascending=False)
            out = []
            for _, row in filtered.head(top_n).iterrows():
                out.append({
                    "title": row[self.COL_TITLE],
                    "year": str(row[self.COL_YEAR]) if pd.notna(row[self.COL_YEAR]) else "",
                    "genre": row[self.COL_GENRE],
                    "poster": row[self.COL_POSTER],
                    "rating": float(row[self.COL_RATING]) if pd.notna(row[self.COL_RATING]) else None
                })
            return out
        else:
            mask = self.df['Overview'].fillna('').str.lower().str.contains(actor_name.lower())
            filtered = self.df[mask]
            out = []
            for _, row in filtered.head(top_n).iterrows():
                out.append({
                    "title": row[self.COL_TITLE],
                    "year": str(row[self.COL_YEAR]) if pd.notna(row[self.COL_YEAR]) else "",
                    "genre": row[self.COL_GENRE],
                    "poster": row[self.COL_POSTER],
                    "rating": float(row[self.COL_RATING]) if pd.notna(row[self.COL_RATING]) else None
                })
            return out

    def surprise_me(self, top_n: int = 4, bias_title: Optional[str] = None) -> List[Dict]:
        if self.COL_RATING in self.df.columns:
            df = self.df.copy()
            df['pop'] = df[self.COL_RATING].fillna(df[self.COL_RATING].median())
            sampled = df.sample(n=min(top_n*3, len(df)), weights='pop', random_state=None)
            sampled = sampled.sample(n=min(top_n, len(sampled)))
        else:
            sampled = self.df.sample(n=min(top_n, len(self.df)))
        if bias_title:
            try:
                bias_recs = self.recommend(bias_title, top_n=1)
                if bias_recs:
                    sampled = pd.concat([pd.DataFrame([bias_recs[0]]), sampled]).drop_duplicates(self.COL_TITLE).head(top_n)
            except Exception:
                pass
        out = []
        for _, row in sampled.head(top_n).iterrows():
            out.append({
                "title": row[self.COL_TITLE],
                "year": str(row[self.COL_YEAR]) if pd.notna(row[self.COL_YEAR]) else "",
                "genre": row[self.COL_GENRE],
                "poster": row[self.COL_POSTER],
                "rating": float(row[self.COL_RATING]) if pd.notna(row[self.COL_RATING]) else None
            })
        return out

    def trailer_search_url(self, movie_title: str) -> str:
        q = f"{movie_title} trailer"
        return f"https://www.youtube.com/results?search_query={q.replace(' ', '+')}"

    # -----------------------------
    # Watchlist
    # -----------------------------
    def load_watchlist(self) -> List[Dict]:
        if self.watchlist_file.exists():
            try:
                return json.loads(self.watchlist_file.read_text(encoding='utf-8'))
            except Exception:
                return []
        return []

    def save_watchlist(self, items: List[Dict]):
        self.watchlist_file.write_text(json.dumps(items, indent=2), encoding='utf-8')

    def add_to_watchlist(self, movie_title: str) -> bool:
        details = None
        if movie_title in self.df[self.COL_TITLE].values:
            row = self.df[self.df[self.COL_TITLE] == movie_title].iloc[0]
            details = {
                "title": row[self.COL_TITLE],
                "year": str(row[self.COL_YEAR]) if pd.notna(row[self.COL_YEAR]) else "",
                "genre": row[self.COL_GENRE],
                "poster": row[self.COL_POSTER]
            }
        else:
            matched, _ = self._fuzzy_match_title(movie_title)
            if matched:
                row = self.df[self.df[self.COL_TITLE] == matched].iloc[0]
                details = {
                    "title": row[self.COL_TITLE],
                    "year": str(row[self.COL_YEAR]) if pd.notna(row[self.COL_YEAR]) else "",
                    "genre": row[self.COL_GENRE],
                    "poster": row[self.COL_POSTER]
                }
        if not details:
            return False
        wl = self.load_watchlist()
        if any(x.get('title') == details['title'] for x in wl):
            return False
        wl.append(details)
        self.save_watchlist(wl)
        return True
