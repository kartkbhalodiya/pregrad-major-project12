#!/usr/bin/env python
import pandas as pd
import os

current_dir = r'D:\pregrad major project'
data_path = os.path.join(current_dir, 'data', 'netflix_sample.csv')
df = pd.read_csv(data_path)

poster_urls = [
    "https://image.tmdb.org/t/p/w500/nz8Vryn6HhWRD8zn7jXoGpSpRqa.jpg",
    "https://image.tmdb.org/t/p/w500/dPnqaFZUL4l0V3pN0b8W6bDiQlS.jpg",
    "https://image.tmdb.org/t/p/w500/qxo8XUlXaZJdE7uYPFr2YH1Aah8.jpg",
    "https://image.tmdb.org/t/p/w500/r12qEDYewehughes.jpg",
    "https://image.tmdb.org/t/p/w500/zaFM8IcsYwN3duKvEmdYszvu98.jpg",
    "https://image.tmdb.org/t/p/w500/x2RS2BlLwi4h9Av2x0k52sPJN6W.jpg",
]

if 'poster_url' not in df.columns:
    df['poster_url'] = [poster_urls[i % len(poster_urls)] for i in range(len(df))]
    df.to_csv(data_path, index=False)
    print("✓ Added poster_url column successfully!")
    print(f"✓ Total records: {len(df)}")
else:
    print("✓ poster_url column already exists!")
    print(f"✓ Total records: {len(df)}")
