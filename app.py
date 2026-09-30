from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load trained AI model
model = joblib.load("models/movie_model.pkl")

movies = model["movies"]
similarity = model["similarity"]


def recommend_movies(movie_name, number=5):

    matches = movies[
        movies["name"].str.lower() == movie_name.strip().lower()
    ]

    if matches.empty:
        return None

    movie_index = matches.index[0]

    scores = list(enumerate(similarity[movie_index]))

    scores = sorted(
        scores,
        key=lambda x: x[1],
        reverse=True
    )

    scores = [
        item for item in scores
        if item[0] != movie_index
    ]

    recommendations = []

    for index, score in scores[:number]:

        movie = movies.iloc[index]

        recommendations.append({
            "name": movie["name"],
            "genre": movie["genre"],
            "year": movie["year"],
            "similarity": round(float(score) * 100, 1)
        })

    return recommendations


@app.route("/", methods=["GET", "POST"])
def home():

    recommendations = []
    searched_movie = ""
    error = ""

    if request.method == "POST":

        searched_movie = request.form.get("movie_name", "").strip()

        if searched_movie:

            recommendations = recommend_movies(searched_movie)

            if recommendations is None:
                error = "Movie not found. Please check the spelling."

        else:
            error = "Please enter a movie name."

    return render_template(
        "index.html",
        recommendations=recommendations,
        searched_movie=searched_movie,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)