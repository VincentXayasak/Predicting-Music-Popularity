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
os.makedirs("box_vs_popularity", exist_ok=True)

# Step 4: Create and save box plots
for feature in features:
    plt.figure(figsize=(10, 6))
    
    if df[feature].nunique() > 10 and df[feature].dtype != 'object':
        # For continuous features with many unique values, bin them
        binned_feature = pd.qcut(df[feature], q=5, duplicates='drop')
        sns.boxplot(x=binned_feature, y=df['popularity'])
        plt.xlabel(f'{feature} (binned)')
    else:
        # For categorical or low-unique features
        sns.boxplot(x=df[feature].astype(str), y=df['popularity'])
        plt.xlabel(feature)

    plt.title(f'{feature} vs Popularity (Box Plot)')
    plt.ylabel('Popularity')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"box_vs_popularity/{feature}_vs_popularity_box.png")
    plt.close()

print("All box plots saved to folder 'box_vs_popularity'.")
