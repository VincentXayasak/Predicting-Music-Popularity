import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

conn = sqlite3.connect("music_data.db")
df = pd.read_sql_query("SELECT * FROM songs", conn)
conn.close()

features = [
    'duration_ms','explicit','danceability', 'energy','key','loudness','mode',
    'speechiness', 'acousticness', 'instrumentalness','liveness','valence',
    'tempo','time_signature'
]

plt.figure(figsize=(15, 15))
for i, feature in enumerate(features, 1):
    plt.subplot(5, 3, i)
    sns.scatterplot(x=df[feature], y=df['popularity'], alpha=0.3, s=10)
    plt.title(f'{feature} vs Popularity')
    plt.xlabel(feature)
    plt.ylabel('Popularity')

plt.tight_layout()
plt.show()