import pandas as pd
import joblib
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("Loading movie dataset...")

# Read dataset
movies = pd.read_csv("dataset/archive/movies_updated.csv")

# Select useful columns
features = ["name", "genre", "director", "writer", "star", "company"]

# Handle missing values
for column in features:
    movies[column] = movies[column].fillna("")

# Combine movie information
movies["combined_features"] = (
    movies["name"] + " " +
    movies["genre"] + " " +
    movies["director"] + " " +
    movies["writer"] + " " +
    movies["star"] + " " +
    movies["company"]
)

print("Converting movie information into numbers...")

# Convert text into numerical vectors
vectorizer = TfidfVectorizer(stop_words="english")

tfidf_matrix = vectorizer.fit_transform(
    movies["combined_features"]
)

print("Calculating movie similarities...")

# Calculate similarity between movies
similarity_matrix = cosine_similarity(tfidf_matrix)

# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save trained model and movie data
joblib.dump(
    {
        "movies": movies,
        "similarity": similarity_matrix
    },
    "models/movie_model.pkl"
)

print("\nAI Model trained successfully!")
print("Total movies:", len(movies))
print("Model saved in models/movie_model.pkl")
