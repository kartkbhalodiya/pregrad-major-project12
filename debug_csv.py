import pandas as pd

df = pd.read_csv(r'd:\pregrad major project\data\netflix_sample.csv', encoding='latin-1')
print('Columns:', df.columns.tolist())
print('\nFirst row:')
print(df.iloc[0])
print('\nDataframe shape:', df.shape)
print('\nData types:')
print(df.dtypes)
