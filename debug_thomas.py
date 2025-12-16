import pandas as pd

try:
    df = pd.read_csv('data/netflix_sample.csv', encoding='latin-1')
    print("Searching for 'Thomas' in titles...")
    thomas_rows = df[df['Title'].str.contains('Thomas', case=False, na=False)]
    
    for idx, row in thomas_rows.iterrows():
        print(f"\nIndex: {idx}")
        print(f"Title: {row['Title']}")
        print(f"Director: {row['Director']}")
        print(f"Genre: {row['Genre']}")
        print(f"Release Date: {row['Release Date']}")
except Exception as e:
    print(f"Error: {e}")
