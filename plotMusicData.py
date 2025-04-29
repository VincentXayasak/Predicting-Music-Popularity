import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Step 1: Connect to SQLite and load the dataset
conn = sqlite3.connect("music_data.db")
df = pd.read_sql_query("SELECT * FROM songs", conn)
conn.close()

# Step 2: List of features to compare against popularity
features = [
    'duration_ms','explicit','danceability', 'energy','key','loudness','mode',
    'speechiness', 'acousticness', 'instrumentalness','liveness','valence',
    'tempo','time_signature'
]

# Step 3: Create output folder
os.makedirs("scatter_vs_popularity", exist_ok=True)

# Step 4: Create and save scatter plots
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