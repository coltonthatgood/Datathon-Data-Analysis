# To run this dashboard, make sure you have streamlit, pandas, matplotlib, and numpy installed.
# Then, save this file as dashboard.py and run the following command in your terminal:
# streamlit run dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import ast  # For safe_literal_eval
import seaborn as sns
from wordcloud import WordCloud

# Note: This dashboard requires a CSV file named 'cleaned_videos.csv' in the same directory.
# If you don't have this file, the app will fail to run.

# Load data
try:
    df = pd.read_csv('data/cleaned_videos.csv')
except FileNotFoundError:
    st.error("Error: 'cleaned_videos.csv' not found. Please ensure the file is in the same directory as this script.")
    st.stop()

# Recreate missing columns (post-load fix)
df['publishedAt'] = pd.to_datetime(df['publishedAt'], errors='coerce')
df.dropna(subset=['publishedAt'], inplace=True)
df['year'] = df['publishedAt'].dt.year

# Recreate tags_parsed if missing
if 'tags_parsed' not in df.columns:
    def safe_literal_eval(x):
        if not isinstance(x, str) or pd.isna(x):
            return []
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return []
    df['tags_parsed'] = df['tags'].apply(safe_literal_eval)
    st.info("Created 'tags_parsed' column from 'tags'.")

# Recreate other keys
if 'topicCategories' in df.columns and 'main_topic' not in df.columns:
    df['topicCategories'] = df['topicCategories'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
    df['main_topic'] = df['topicCategories'].apply(lambda x: x[0] if x else 'Unknown')
else:
    st.info("Using 'main_topic' column if available, otherwise defaulting to 'Unknown'.")
    if 'main_topic' not in df.columns:
        df['main_topic'] = 'Unknown'

# Filter out rows with missing 'viewCount' or 'likeCount' to avoid errors
df.dropna(subset=['viewCount', 'likeCount'], inplace=True)
df['is_genz_proxy'] = ((df['contentDuration'] < 60) & (df['year'] >= 2023) & (df['likeCount'] / (df['viewCount'] + 1) > 0.05)).astype(int)

st.title('Beauty Trends Dashboard: Gen Z Chases')

# Sidebar filters
available_years = sorted(df['year'].unique())
if not available_years:
    st.error("No years found in the dataset.")
    st.stop()
year = st.sidebar.selectbox('Year', available_years)

top_topics = df['main_topic'].value_counts().head(5).index.tolist()
if not top_topics:
    st.error("No topics found in the dataset.")
    st.stop()
topic = st.sidebar.selectbox('Topic', top_topics)

st.subheader(f'Trends in {year} for Topic: {topic}')
df_filtered = df[(df['year'] == year) & (df['main_topic'] == topic)].copy()

if df_filtered.empty:
    st.warning("No data available for the selected year and topic.")
else:
    # Plot 1: Keyword Mentions Over Time
    st.subheader('Keyword Mentions Over Time')
    fig1, ax1 = plt.subplots()
    if 'tags_parsed' in df_filtered.columns:
        df_filtered['has_makeup'] = df_filtered['tags_parsed'].apply(lambda tags: any('makeup' in tag.lower() for tag in tags if tags))
        df_filtered.groupby(df_filtered['publishedAt'].dt.month)['has_makeup'].sum().plot(ax=ax1, title='Makeup Mentions by Month')
    else:
        st.warning("No 'tags_parsed' column available to plot keyword mentions.")
    st.pyplot(fig1)

    # Plot 2: Views by Sentiment (fallback if not computed)
    st.subheader('Views by Sentiment')
    if 'sentiment_bin' not in df_filtered.columns:
        df_filtered['overall_sentiment'] = 0.1  # Fallback avg
        df_filtered['sentiment_bin'] = pd.cut(df_filtered['overall_sentiment'], bins=3, labels=['Negative', 'Neutral', 'Positive'])
    
    sentiment_data = df_filtered.groupby('sentiment_bin', observed=True)['viewCount'].mean()
    fig2, ax2 = plt.subplots()
    sentiment_data.plot(kind='bar', ax=ax2)
    st.pyplot(fig2)

    # Plot 3: Top Gen Z Products (safe explode)
    st.subheader('Top Gen Z Products')
    genz_videos = df_filtered[df_filtered['is_genz_proxy'] == 1]
    genz_tags = []
    if 'tags_parsed' in genz_videos.columns:
        for tags in genz_videos['tags_parsed']:
            genz_tags.extend([tag.lower() for tag in tags if isinstance(tag, str)])
    genz_counter = Counter(genz_tags)
    
    if genz_counter:
        fig3, ax3 = plt.subplots()
        pd.Series(genz_counter).head(10).plot(kind='barh', ax=ax3, title='Top Gen Z Tags')
        st.pyplot(fig3)
    else:
        st.info("No Gen Z proxy videos found for the selected filters.")


    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Views", f"{df_filtered['viewCount'].mean():,.0f}")
    col2.metric("Gen Z Videos %", f"{(len(genz_videos) / len(df_filtered)) * 100:.2f}%")
    col3.metric("Top Keyword Mentions", genz_counter.most_common(1)[0][1] if genz_counter else 0)

# Load comments data

# Load dataset
@st.cache_data
def load_data():
    df1 = pd.read_csv("data/cleaned_comments_with_features.csv")
    df1['publishedAt'] = pd.to_datetime(df1['publishedAt'], errors='coerce')
    df1['year'] = df1['publishedAt'].dt.year
    df1['month'] = df1['publishedAt'].dt.to_period("M").astype(str)
    return df1

df1 = load_data()

st.title("YouTube Comments Dashboard")

# Sidebar filters
years = sorted(df1['year'].dropna().unique())
year_filter = st.sidebar.multiselect("Select Year(s)", years, default=years)

df1_filtered = df1[df1['year'].isin(year_filter)]

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Trends", "Engagement", "Sentiment & Text"])

#Overview Metrics
with tab1:
    st.header("Overview Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Comments", f"{len(df1_filtered):,}")
    col2.metric("Avg Likes per Comment", f"{df1_filtered['likeCount'].mean():.2f}")
    col3.metric("Unique Authors", f"{df1_filtered['authorId'].nunique():,}")

    st.subheader("Sentiment Distribution")
    sentiment_counts = df1_filtered['sentiment'].value_counts()
    st.bar_chart(sentiment_counts)

#Trends Metrics
with tab2:
    st.header("Trends Over Time")

    st.subheader("Comments per Month")
    monthly = df1_filtered.groupby("month").size()
    st.line_chart(monthly)

    st.subheader("Average Likes per Month")
    monthly_likes = df1_filtered.groupby("month")['likeCount'].mean()
    st.line_chart(monthly_likes)

    st.subheader("Comment Activity Heatmap (Weekday vs Hour)")
    heatmap_data = df1_filtered.groupby(["comment_weekday","comment_hour"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(10,5))
    sns.heatmap(heatmap_data, cmap="Blues", ax=ax)
    st.pyplot(fig)

# Engagement Metrics
with tab3:
    st.header("Engagement Analysis")

    st.subheader("Top 10 Comments by Likes")
    top_comments = df1_filtered[['textOriginal','likeCount']].sort_values(by="likeCount", ascending=False).head(10)
    st.table(top_comments)

    st.subheader("Most Active Authors")
    top_authors = df1_filtered['authorId'].value_counts().head(10)
    st.bar_chart(top_authors)

    st.subheader("Replies vs. Original Comments")
    reply_split = df1_filtered['is_reply'].value_counts()
    st.bar_chart(reply_split)

# Sentiment & Text Features
with tab4:
    st.header("Sentiment & Text Features")

    st.subheader("Sentiment vs Likes")
    sentiment_likes = df1_filtered.groupby("sentiment")['likeCount'].mean()
    st.bar_chart(sentiment_likes)

    st.subheader("Word Cloud of Keywords")
    text = " ".join(df1_filtered['keywords'].dropna().astype(str))
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    fig, ax = plt.subplots(figsize=(10,5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

    st.subheader("Average Text Length & Emoji Density Over Time")
    fig, ax = plt.subplots(figsize=(10,5))
    df1_filtered.groupby("month")[['textlen','emoji_density']].mean().plot(ax=ax)
    st.pyplot(fig)
