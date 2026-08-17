import pandas as pd
df = pd.read_csv('../data/bbc_data.csv')

print("##### count of rows/columns")
print(df.shape)
print("##### column names")
print(df.columns)
##### df head
print(df.head())
##### data types
print(df.info())
##### balance of labels (5 classes)
print(df['labels'].value_counts())
##### dirty texts
print(df['data'].iloc[0])
##### length of texts
print(df['data'].str.split().str.len().describe())
##### min/max/mean/std
print(df.describe())
##### sum of duplicates
print(df.duplicated().sum())
