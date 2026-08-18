import pandas as pd
import re
from sklearn.model_selection import train_test_split

def extract(file):
    df = pd.read_csv(file)
    df = df.drop_duplicates()
    return df

def transform(text):
    text = re.sub(r'(x[0-9a-f]{2}){1,2}', '', text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load(path):
    df = extract(path)
    df['data'] = df['data'].apply(transform)
    df['labels'] = pd.factorize(df['labels'])[0]
    X = df['data']
    y = df['labels']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test




