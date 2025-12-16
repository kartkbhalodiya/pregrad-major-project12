import pandas as pd
import numpy as np

def clean_thomas_data():
    csv_path = 'data/netflix_sample.csv'
    print(f"Reading {csv_path}...")
    
    try:
        df = pd.read_csv(csv_path, encoding='latin-1')
        
        # Identify the rows to fix
        mask = (df['Title'].str.contains('Thomas & Friends', case=False, na=False)) & \
               (df['Director'] == 'Miika Soini')
        
        rows_to_fix = df[mask]
        print(f"Found {len(rows_to_fix)} rows to fix.")
        
        if len(rows_to_fix) == 0:
            print("No rows found matching the criteria. Data might already be clean.")
            return

        # Define correct data
        correct_data = {
            'Director': 'Joey So',
            'Genre': 'Children & Family Movies, Animation',
            'Actors': 'Joseph May, Marie Ekins, Rob Rackstraw',
            'Country Availability': 'United Kingdom',
            'Release Date': '01-May-20' # Approximate date for 2020 specials
        }
        
        # Apply fixes
        for idx in rows_to_fix.index:
            print(f"Fixing row {idx}: {df.at[idx, 'Title']}")
            df.at[idx, 'Director'] = correct_data['Director']
            df.at[idx, 'Genre'] = correct_data['Genre']
            df.at[idx, 'Actors'] = correct_data['Actors']
            df.at[idx, 'Country Availability'] = correct_data['Country Availability']
            df.at[idx, 'Release Date'] = correct_data['Release Date']
            
            # Also fix specific cast for Royal Engine if possible
            if 'Royal Engine' in df.at[idx, 'Title']:
                df.at[idx, 'Actors'] = 'Rosamund Pike, Joseph May, Rob Rackstraw'

        # Save back to CSV
        print("Saving corrected data...")
        df.to_csv(csv_path, index=False, encoding='latin-1')
        print("Done!")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    clean_thomas_data()
