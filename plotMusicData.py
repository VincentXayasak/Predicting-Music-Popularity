import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv("music_dataset.csv")

features = [
    'duration_ms','explicit','danceability', 'energy','key','loudness','mode',
    'speechiness', 'acousticness', 'instrumentalness','liveness','valence',
    'tempo','time_signature'
]

os.makedirs("scatter_vs_popularity", exist_ok=True)

for feature in features:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df[feature], y=df['popularity'], alpha=0.3, s=10)
    plt.title(f'{feature} vs Popularity')
    plt.xlabel(feature)
    plt.ylabel('Popularity')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"scatter_vs_popularity/{feature}_vs_popularity.png")
    plt.close()

print("All scatter plots saved to folder 'scatter_vs_popularity'.")