import pandas as pd
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import numpy as np
import re

def analyze_sentiments(csv_path, output_csv="analyzed_tweets.csv"):
    analyzer = SentimentIntensityAnalyzer()

    # Football-specific lexicon boost
    football_lexicon = {
        'goal': 4.0, 'miss': -3.5, 'foul': -2.7, 'save': 3.2, 'tackle': 1.5, 'dive': -2.0,
        'penalty': -1.8, 'shot on target': 2.5, 'what a goal': 4.5, 'horrible miss': -4.0,
        'clinical finish': 4.2, 'clear foul': -3.0, 'wasted chance': -3.5, 'brilliant save': 4.0,
        'amazing tackle': 2.5, 'perfect through ball': 3.2, 'awful defensive error': -4.0
    }
    analyzer.lexicon.update(football_lexicon)

    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'#\w+', '', text)  # Remove hashtags
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        return text.strip()

    def analyze_football_sentiment(tweet, team1, team2):
        if not isinstance(tweet, str):
            return 'neutral'  # Skip non-string tweets
        
        tweet_cleaned = tweet.lower().replace(team1.lower(), '').replace(team2.lower(), '')
        scores = analyzer.polarity_scores(tweet_cleaned)

        # Football contextual adjustments
        context_adjustments = {
            'goal': 0.25, 'miss': -0.3, 'foul': -0.2, 'save': 0.2,
            'brilliant save': 0.5, 'clear foul': -0.3, 'clinical finish': 0.3,
            'awful defensive error': -0.4, 'perfect through ball': 0.4
        }
        for keyword, adjust in context_adjustments.items():
            if keyword in tweet_cleaned:
                scores['compound'] += adjust

        if scores['compound'] >= 0.05:
            return 'positive'
        elif scores['compound'] <= -0.05:
            return 'negative'
        return 'neutral'

    df = pd.read_csv(csv_path)

    # Validate structure
    required_cols = {'timestamp', 'tweet', 'team'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    df.dropna(subset=['tweet', 'team'], inplace=True)  # Robustness against missing values
    df['tweet'] = df['tweet'].astype(str)

    # Extract unique teams
    teams = df['team'].dropna().unique()
    if len(teams) != 2:
        raise ValueError(f"Exactly 2 teams required, found {len(teams)}: {teams}")

    # Apply analysis
    df['sentiment'] = df.apply(
        lambda row: analyze_football_sentiment(row['tweet'], teams[0], teams[1]), axis=1
    )

    df.to_csv(output_csv, index=False)
    return df

def visualize_sentiments(csv_path):
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df.dropna(subset=['timestamp', 'sentiment', 'team'], inplace=True)

    df['time_bin'] = df['timestamp'].dt.floor('5min')
    teams = df['team'].unique()
    if len(teams) != 2:
        raise ValueError("Visualization requires exactly 2 teams")

    df['sentiment_score'] = df['sentiment'].map({'positive': 1, 'negative': -1, 'neutral': 0})
    df['cumulative_sentiment'] = df.groupby('team')['sentiment_score'].transform(lambda x: x.cumsum())

    pivot_df = df.pivot_table(index='time_bin', columns='team', values='cumulative_sentiment', aggfunc='last').reset_index().ffill()

    def softmax_prob(row):
        scores = row[teams].values
        if np.isnan(scores).any():
            return [50, 50]
        exp_scores = np.exp(scores - np.max(scores))
        return exp_scores / exp_scores.sum() * 100

    probabilities = pivot_df[teams].apply(softmax_prob, axis=1, result_type='expand')
    probabilities.columns = teams
    pivot_df = pd.concat([pivot_df[['time_bin']], probabilities], axis=1)

    plot_df = pivot_df.melt(id_vars='time_bin', var_name='team', value_name='probability')

    plt.figure(figsize=(12, 6))
    for team in teams:
        team_df = plot_df[plot_df['team'] == team]
        plt.plot(team_df['time_bin'], team_df['probability'], label=team, marker='o', linestyle='-')

    plt.title('Real-Time Winning Probability Estimation')
    plt.xlabel('Match Time')
    plt.ylabel('Winning Probability (%)')
    plt.xticks(rotation=45)
    plt.ylim(0, 100)
    plt.axhline(50, color='gray', linestyle='--', alpha=0.5)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    input_csv = "der_klassiker.csv"
    output_csv = "analyzed_results.csv"
    try:
        analyzed_df = analyze_sentiments(input_csv, output_csv)
        visualize_sentiments(output_csv)
    except Exception as e:
        print("❌ Error:", e)
    else:
        print("✅ Sentiment analysis and visualization completed successfully.")
