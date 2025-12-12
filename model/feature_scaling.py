import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import MultiLabelBinarizer

def preprocess_data(df):
    df = df.copy()
    
    y = (df['$Worldwide'])
    y = np.log1p(y)

    df['Genres'] = df['Genres'].apply(ast.literal_eval)
    df['Countries'] = df['Production_Countries'].apply(ast.literal_eval)
    df['Social_Buzz'] = df['Rating'] * df['Vote_Count']

    # one hot encoding
    mlb_genres = MultiLabelBinarizer()
    genres_encoded = mlb_genres.fit_transform(df['Genres'])
    genres_df = pd.DataFrame(genres_encoded, columns=[f"Genre_{c}" for c in mlb_genres.classes_], index=df.index)

    mlb_countries = MultiLabelBinarizer()
    countries_encoded = mlb_countries.fit_transform(df['Countries'])
    countries_df = pd.DataFrame(countries_encoded, columns=[f"Country_{c}" for c in mlb_countries.classes_], index=df.index)

    X_numeric = df[['Year', 'Rating', 'Vote_Count','Social_Buzz',]]
    
    X = pd.concat([X_numeric, genres_df, countries_df], axis=1)
    
    return X, y