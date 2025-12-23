import matplotlib
matplotlib.use("Agg")

import os, uuid, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- LOAD DATA ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "candidate_dataset_enriched.csv"))

# ---------------- CLEAN EXPERIENCE ----------------
def extract_years(x):
    match = re.search(r"[\d.]+", str(x))
    return float(match.group()) if match else 0.0

df["exp_years"] = df["experience"].apply(extract_years)

# ---------------- TEXT COMBINATION ----------------
df["combined_text"] = (
    df["domain"].fillna("") + " " +
    df["skills"].fillna("") + " " +
    df["projects"].fillna("")
).str.lower()

# ---------------- VECTORIZERS ----------------
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["combined_text"])

bow = CountVectorizer(stop_words="english")
bow_matrix = bow.fit_transform(df["combined_text"])

# ---------------- NORMALIZE SCORE ----------------
def normalize(scores):
    scores = np.array(scores, dtype=float)
    if scores.max() == scores.min():
        return np.ones(len(scores))
    return (scores - scores.min()) / (scores.max() - scores.min())

# ---------------- RECOMMENDER ----------------
def recommend(job_desc, top_n=5, model="tfidf", show_graph=True, save_graph=True):

    job_desc = job_desc.lower()

    # 1️⃣ TF-IDF → semantic similarity
    if model == "tfidf":
        job_vec = tfidf.transform([job_desc])
        scores = cosine_similarity(job_vec, tfidf_matrix)[0]

    # 2️⃣ Bag of Words → keyword frequency
    elif model == "bow":
        job_vec = bow.transform([job_desc])
        scores = (bow_matrix @ job_vec.T).toarray().ravel()

    # 3️⃣ Random Forest (SIMULATED) → experience-weighted relevance
    elif model == "rf":
        job_vec = tfidf.transform([job_desc])
        sim = cosine_similarity(job_vec, tfidf_matrix)[0]
        scores = sim * (1 + df["exp_years"] / 5)

    # 4️⃣ Naive Bayes (SIMULATED) → skill overlap probability
    elif model == "nb":
        job_words = set(job_desc.split())
        scores = df["skills"].apply(
            lambda s: len(job_words & set(str(s).lower().split()))
        )

    # 5️⃣ Decision Tree (RULE-BASED)
    elif model == "dt":
        scores = []
        for _, row in df.iterrows():
            score = 0
            if row["exp_years"] >= 3:
                score += 2
            if "python" in row["skills"].lower():
                score += 2
            if "react" in row["skills"].lower():
                score += 1
            scores.append(score)
        scores = np.array(scores)

    else:
        scores = cosine_similarity(
            tfidf.transform([job_desc]), tfidf_matrix
        )[0]

    # ✅ NORMALIZE
    scores = normalize(scores)

    # ---------------- SORT ----------------
    result = df.copy()
    result["match_score"] = scores
    result = result.sort_values("match_score", ascending=False).head(top_n)

    # ---------------- GRAPH ----------------
    graph_path = None
    if show_graph:
        plt.figure(figsize=(7, 4))
        plt.bar(result["Candidate Name"], result["match_score"])
        plt.xticks(rotation=30)
        plt.title(f"Top Candidates ({model.upper()})")
        plt.tight_layout()

        fname = f"graph_{uuid.uuid4().hex}.png"
        graph_path = f"static/{fname}"
        plt.savefig(os.path.join(BASE_DIR, graph_path))
        plt.close()

    # ---------------- FINAL OUTPUT ----------------
    final = result[[
        "Candidate Name",
        "domain",
        "skills",
        "experience",
        "projects",
        "match_score"
    ]].rename(columns={
        "Candidate Name": "name"
    }).to_dict(orient="records")

    return final, graph_path, {}
