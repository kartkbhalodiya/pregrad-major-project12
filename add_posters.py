import pandas as pd
import os

data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'netflix_sample.csv')
df = pd.read_csv(data_path)

poster_urls = [
    "https://image.tmdb.org/t/p/w500/nz8Vryn6HhWRD8zn7jXoGpSpRqa.jpg",
    "https://image.tmdb.org/t/p/w500/dPnqaFZUL4l0V3pN0b8W6bDiQlS.jpg",
    "https://image.tmdb.org/t/p/w500/qxo8XUlXaZJdE7uYPFr2YH1Aah8.jpg",
    "https://image.tmdb.org/t/p/w500/r12qEDYewehughes.jpg",
    "https://image.tmdb.org/t/p/w500/zaFM8IcsYwN3duKvEmdYszvu98.jpg",
    "https://image.tmdb.org/t/p/w500/x2RS2BlLwi4h9Av2x0k52sPJN6W.jpg",
    "https://image.tmdb.org/t/p/w500/iajEHXn9Freal.jpg",
    "https://image.tmdb.org/t/p/w500/kOTRj2ghEpVXXXo.jpg",
]

if 'poster_url' not in df.columns:
    df['poster_url'] = df.index % len(poster_urls)
    df['poster_url'] = df['poster_url'].apply(lambda x: poster_urls[x])
    df.to_csv(data_path, index=False)
    print("Added poster_url column successfully!")
    print(f"Total records: {len(df)}")
else:
    print("poster_url column already exists")
    print(f"Total records: {len(df)}")
