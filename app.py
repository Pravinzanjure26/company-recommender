from flask import Flask, render_template, request
from recommender import recommend

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []
    graph_path = None
    model = "tfidf"
    top_n = 5

    if request.method == "POST":
        job = request.form["job_description"]
        model = request.form["model"]
        top_n = int(request.form["top_n"])

        recommendations, graph_path, _ = recommend(job, top_n, model)

    return render_template(
        "index.html",
        recommendations=recommendations,
        graph_path=graph_path,
        model=model,
        top_n=top_n
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

