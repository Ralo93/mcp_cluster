import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import re

from data_loader import DataLoader


def visualize_overview(df):
    if df.empty:
        print("❌ DataFrame is empty. No data to visualize.")
        return

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    sns.set(style="whitegrid")

    # Distribution by Faction
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='faction', order=df['faction'].value_counts().index)
    plt.title('Distribution of Speeches by Faction')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Time Distribution
    plt.figure(figsize=(12, 6))
    df.set_index('date')['faction'].resample('M').count().plot()
    plt.title('Number of Speeches Over Time')
    plt.ylabel('Speech Count')
    plt.xlabel('Date')
    plt.tight_layout()
    plt.show()

    # Cluster Distribution
    if 'cluster' in df.columns and df['cluster'].notna().any():
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x='cluster', order=df['cluster'].value_counts().index)
        plt.title('Distribution of Speeches by Cluster')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

def analyze_content(df):
    if 'content' not in df.columns or df['content'].dropna().empty:
        print("❌ No content available to analyze.")
        return

    content_clean = df['content'].dropna().apply(lambda x: re.sub(r'[^a-zA-ZäöüÄÖÜß\\s]', '', x.lower()))
    word_list = sum([text.split() for text in content_clean], [])
    word_counts = Counter(word_list)
    total_words = sum(word_counts.values())
    unique_words = len(word_counts)

    print(f"📊 Total Speeches: {len(df)}")
    print(f"📝 Total Words: {total_words}")
    print(f"🔤 Unique Words: {unique_words}")
    print("📈 Top 10 Words:")
    print(word_counts.most_common(10))

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_counts)
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Most Common Words in Speech Content')
    plt.tight_layout()
    plt.show()

def main():
    loader = DataLoader()
    df = loader.load_from_db()
    if df is not None:
        visualize_overview(df)
        analyze_content(df)

if __name__ == "__main__":
    main()