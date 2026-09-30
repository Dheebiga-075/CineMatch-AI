import joblib

# Load the trained AI model
print("Loading CineMatch AI model...")

model = joblib.load("models/movie_model.pkl")

movies = model["movies"]
similarity = model["similarity"]

print("Model loaded successfully!")

def recommend_movies(movie_name, number=5):

    # Search for the movie
    matches = movies[
        movies["name"].str.lower() == movie_name.strip().lower()
    ]

    if matches.empty:
        print("\nMovie not found!")
        print("Please check the spelling or try another movie.")
        return

    # Get the movie index
    movie_index = matches.index[0]

    # Get similarity scores
    scores = list(enumerate(similarity[movie_index]))

    # Sort from most similar to least similar
    scores = sorted(
        scores,
        key=lambda x: x[1],
        reverse=True
    )

    # Remove the selected movie itself
    scores = [
        item for item in scores
        if item[0] != movie_index
    ]

    # Display recommendations
    print(f"\nMovies similar to '{movies.iloc[movie_index]['name']}':\n")

    for rank, (index, score) in enumerate(scores[:number], start=1):

        movie = movies.iloc[index]

        print(f"{rank}. {movie['name']}")
        print(f"   Genre: {movie['genre']}")
        print(f"   Year: {movie['year']}")
        print(f"   Similarity: {score:.2f}")
        print()

# User input
movie_name = input("Enter a movie name: ")

recommend_movies(movie_name)
