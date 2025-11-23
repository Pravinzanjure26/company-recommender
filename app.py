from flask import Flask, render_template, request
from recommender import recommend

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    query = ""
    recommendations = []

    if request.method == "POST":
        query = request.form.get("job_description", "")
        if query.strip():
            recommendations = recommend(query, top_n=5)

    return render_template("index.html", query=query, recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)
