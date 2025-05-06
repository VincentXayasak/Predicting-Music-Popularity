import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image
import glob
import math

# Step 1: Connect to SQLite and load the dataset
conn = sqlite3.connect("music_data.db")
df = pd.read_sql_query("SELECT * FROM songs", conn)
conn.close()

#find and remove songs with zero popularity
zeros = (df['popularity'] == 0).sum()
print(f"zeros: {zeros}")
df = df[df['popularity'] > 0]

# Step 2: List of features to compare against popularity
features = [
    'duration_ms','explicit','danceability', 'energy','key','loudness','mode',
    'speechiness', 'acousticness', 'instrumentalness','liveness','valence',
    'tempo','time_signature'
]

features1 = ['duration_ms','explicit','danceability', 'energy']

feature2 = ['key','loudness','mode',
    'speechiness']

# Step 3: Create output folders
os.makedirs("box_vs_popularity", exist_ok=True)
os.makedirs("histograms", exist_ok=True)
os.makedirs("scatter_plots", exist_ok=True)

# Step 4: Box plots
for feature in features:
    plt.figure(figsize=(10, 6))
    if df[feature].nunique() > 10 and df[feature].dtype != 'object':
        binned_feature = pd.qcut(df[feature], q=1, duplicates='drop')
        sns.boxplot(x=binned_feature, y=df['popularity'])
        plt.xlabel(f'{feature}')
    else:
        sns.boxplot(x=df[feature].astype(str), y=df['popularity'])
        plt.xlabel(feature)
    plt.title(f'{feature} vs Popularity (Box Plot)')
    plt.ylabel('Popularity')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"box_vs_popularity/{feature}_vs_popularity_box.png")
    plt.close()

#heatmap
correlation_matrix = df.corr(numeric_only = True)
plt.figure(figsize = (12, 6))
sns.heatmap(correlation_matrix, annot = True, fmt = '.2f', linewidths = 0.5, cmap = 'coolwarm')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.close()

#histogram
for feature in features + ['popularity']:
    plt.figure(figsize=(8, 5))
    sns.histplot(df[feature], bins = 30, kde = True)
    plt.title(f'{feature} Distribution')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(f"histograms/{feature}_histogram.png")
    plt.close()

#scatterplot
for feature in features:
    plt.figure(figsize = (8, 5))
    sns.scatterplot(x=df[feature], y = df['popularity'])
    plt.title(f'{feature} vs Popularity (Scatter Plot)')
    plt.xlabel(feature)
    plt.ylabel('Popularity')
    plt.tight_layout()
    plt.savefig(f"scatter_plots/{feature}_vs_popularity_scatter.png")
    plt.close()

#everything in one pic
def create_image_collage(image_paths, output_path, images_per_row=4):
    if not image_paths:
        print(f"No images found for {output_path}")
        return

    images = [Image.open(path) for path in image_paths]
    img_width, img_height = images[0].size
    num_images = len(images)
    rows = math.ceil(num_images / images_per_row)

    collage = Image.new('RGB', (images_per_row * img_width, rows * img_height), color='white')

    for i, img in enumerate(images):
        x = (i % images_per_row) * img_width
        y = (i // images_per_row) * img_height
        collage.paste(img, (x, y))

    collage.save(output_path)

create_image_collage(glob.glob("box_vs_popularity/*.png"), "box_collage.png")
create_image_collage(glob.glob("histograms/*.png"), "histogram_collage.png")
create_image_collage(glob.glob("scatter_plots/*.png"), "scatter_collage.png")

#table of the stuff
desc = df[features + ['popularity']].describe().T
formatted_values = desc.apply(lambda col: col.map(lambda x: f"{x:.2e}" if abs(x) >= 100000 else f"{x:.4g}"))

fig, ax = plt.subplots(figsize = (12, len(formatted_values) * 0.5 + 1))
ax.axis('off')
tbl = plt.table(cellText = formatted_values.values,
                colLabels = formatted_values.columns,
                rowLabels = formatted_values.index,
                loc = 'center',
                cellLoc = 'center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.2, 1.2)
plt.title("Summary Statistics Table", fontsize = 14)
plt.tight_layout()
plt.savefig("summary_statistics_table.png")
plt.close()

#missing value check
missing_values = df[features + ['popularity']].isnull().sum().to_frame(name = 'Missing Count')
missing_values['Missing %'] = (missing_values['Missing Count'] / len(df)) * 100
missing_values = missing_values[missing_values['Missing Count'] > 0]  

if not missing_values.empty:
    fig, ax = plt.subplots(figsize = (8, len(missing_values) * 0.5 + 1))
    ax.axis('off')
    tbl = plt.table(cellText = missing_values.round(2).values,
                    colLabels = missing_values.columns,
                    rowLabels = missing_values.index,
                    loc = 'center',
                    cellLoc = 'center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.2)
    plt.title("Missing Values per Feature", fontsize = 14)
    plt.tight_layout()
    plt.savefig("missing_values_table.png")
    plt.close()
    print("Missing values table saved as image.")
else:
    print("No missing values found.")

#popularity artist
if 'artists' in df.columns and df['artists'].notna().any():
    top_artists = df.groupby('artists')['popularity'].mean().sort_values(ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_artists.values, y=top_artists.index)
    plt.xlabel('Average Popularity')
    plt.title('Top 10 Artists by Average Popularity')
    plt.tight_layout()
    plt.savefig("top_artists_avg_popularity.png")
    plt.close()
else:
    print("'artist_name' column not found or contains only missing values.")

#popularity genre
if 'track_genre' in df.columns and df['track_genre'].notna().any():
    top_genres = df.groupby('track_genre')['popularity'].mean().sort_values(ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_genres.values, y=top_genres.index)
    plt.xlabel('Average Popularity')
    plt.title('Top 10 Genres by Average Popularity')
    plt.tight_layout()
    plt.savefig("top_genres_avg_popularity.png")
    plt.close()
else:
    print("'track_genre' column not found or contains only missing values.")


print("bawk")
