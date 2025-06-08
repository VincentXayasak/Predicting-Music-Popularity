import pandas as pd

df = pd.read_csv("UnseenData.csv")

model_features = [
    'duration_ms', 'explicit', 'danceability', 'energy', 'key', 'loudness',
    'mode', 'speechiness', 'acousticness', 'instrumentalness',
    'liveness', 'valence', 'tempo', 'time_signature', 'track_genre'
]

selected_df = df[['track_name', 'popularity'] + model_features]

sampled_df = selected_df.sample(n=10) # Change how many random songs

sampled_df[model_features].to_csv("Random_Songs_Input.csv", index=False)

for idx, row in sampled_df.iterrows():
    print(f"Song: {row['track_name']}, Popularity: {row['popularity']}")