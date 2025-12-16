import chardet
import sys

csv_path = r'd:\pregrad major project\data\netflix_sample.csv'

with open(csv_path, 'rb') as f:
    raw = f.read(100000)
    result = chardet.detect(raw)
    print('Detected encoding:', result)
    
try:
    import pandas as pd
    df = pd.read_csv(csv_path, encoding=result['encoding'])
    print('Successfully read with detected encoding!')
except Exception as e:
    print(f'Failed with detected encoding: {e}')
    
    encodings_to_try = ['latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
    for enc in encodings_to_try:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            print(f'Successfully read with encoding: {enc}')
            break
        except Exception as e2:
            print(f'Failed with {enc}: {str(e2)[:50]}')
