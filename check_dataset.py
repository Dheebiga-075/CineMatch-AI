import pandas as pd

print("CineMatch AI Project Started!")

movies = pd.read_csv("dataset/archive/movies_updated.csv")

print("\nFirst five movies:")
print(movies.head())

print("\nDataset Columns:")
print(movies.columns.tolist())

print("\nTotal Movies:", len(movies))

print("\nMissing Values:")
print(movies.isnull().sum())
