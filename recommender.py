import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Path to enriched CSV
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "candidate_dataset_enriched.csv")

# Load dataset once
df = pd.read_csv(DATA_PATH)

# Build combined text for matching
df["combined_text"] = (
    df["skills"].fillna("") + " " +
    df["domain"].fillna("") + " " +
    df["experience"].fillna("") + " " +
    df["projects"].fillna("")
).str.lower()

# Common replacements
REPLACEMENTS = {
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "tableu": "tableau",
    "js": "javascript",
}

for key, val in REPLACEMENTS.items():
    df["combined_text"] = df["combined_text"].str.replace(key, val)

# Build TF-IDF model
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["combined_text"])


def recommend(job_description: str, top_n: int = 5):

    if not job_description.strip():
        return []

    job_desc = job_description.lower().replace(",", " ")

    for key, val in REPLACEMENTS.items():
        job_desc = job_desc.replace(key, val)

    job_vec = vectorizer.transform([job_desc])
    similarity_scores = cosine_similarity(job_vec, tfidf_matrix)[0]

    result_df = df.copy()
    result_df["match_score"] = similarity_scores

    result_df = result_df.sort_values("match_score", ascending=False).head(top_n)

    return result_df[[
        "Candidate Name",
        "Gender",
        "Age",
        "Email",
        "domain",
        "skills",
        "experience",
        "match_score"
    ]].rename(columns={
        "Candidate Name": "name",
        "Gender": "gender",
        "Age": "age",
        "Email": "email"
    }).to_dict(orient="records")
