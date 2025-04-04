import pandas as pd
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import numpy as np

def analyze_sentiments(csv_path, output_csv="analyzed_tweets.csv"):
    # Initialize analyzer with football-specific adjustments
    analyzer = SentimentIntensityAnalyzer()
    
    # Add football-specific lexicon adjustments
    football_lexicon = {
        'goal': 4.0,          # Very positive
        'miss': -3.5,         # Very negative
        'foul': -2.7,         # Negative
        'save': 3.2,          # Positive
        'tackle': 1.5,        # Mildly positive
        'dive': -2.0,         # Negative
        'penalty': -1.8,      # Contextual negative
        'shot on target': 2.5, # Positive
        'what a goal': 4.5,
        'horrible miss': -4.0,
        'clinical finish': 4.2,
        'clear foul': -3.0,
        'wasted chance': -3.5,
        'brilliant save': 4.0
    }
    
    analyzer.lexicon.update(football_lexicon)

    def analyze_football_sentiment(tweet, team1, team2):
        # Remove team names from analysis
        neutral_tweet = tweet.replace(team1, '').replace(team2, '')
        
        # Get base sentiment
        scores = analyzer.polarity_scores(neutral_tweet)
        
        # Football-specific adjustments
        if 'goal' in tweet.lower():
            scores['compound'] += 0.25
        if 'miss' in tweet.lower():
            scores['compound'] -= 0.3
        if 'foul' in tweet.lower():
            scores['compound'] -= 0.2
            
        # Final classification
        if scores['compound'] >= 0.05:
            return 'positive'
        elif scores['compound'] <= -0.05:
            return 'negative'
        return 'neutral'

    # Load dataset
    df = pd.read_csv(csv_path)
    
    # Get unique teams from the dataset
    teams = df['team'].unique()
    if len(teams) != 2:
        raise ValueError("Dataset must contain exactly 2 teams")

    # Apply enhanced sentiment analysis
    df['sentiment'] = df.apply(
        lambda row: analyze_football_sentiment(row['tweet'], teams[0], teams[1]), 
        axis=1
    )
    
    df.to_csv(output_csv, index=False)
    return df

def visualize_sentiments(csv_path):
    # Load analyzed data
    df = pd.read_csv(csv_path)
    
    # Convert timestamp and create time bins
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['time_bin'] = df['timestamp'].dt.floor('5min')
    
    # Check for exactly 2 teams
    teams = df['team'].unique()
    if len(teams) != 2:
        raise ValueError("Analysis requires exactly 2 teams")

    # Create numerical sentiment scores first
    df['sentiment_score'] = df['sentiment'].map({'positive': 1, 'negative': -1, 'neutral': 0})
    
    # Calculate cumulative scores using transform
    df['cumulative_sentiment'] = df.groupby('team')['sentiment_score'].transform(lambda x: x.cumsum())

    # Create pivot table with forward filling
    pivot_df = df.pivot_table(
        index='time_bin',
        columns='team',
        values='cumulative_sentiment',
        aggfunc='last'
    ).reset_index().ffill()

    # Calculate probabilities using softmax
    def calculate_probabilities(row):
        scores = row[teams].values
        if np.isnan(scores).any():
            return [50, 50]  # Default to equal probability if missing data
        exps = np.exp(scores - np.max(scores))
        return exps / exps.sum() * 100

    # Apply probability calculation
    probabilities = pivot_df[teams].apply(calculate_probabilities, axis=1, result_type='expand')
    probabilities.columns = teams
    pivot_df = pd.concat([pivot_df[['time_bin']], probabilities], axis=1)

    # Melt for plotting
    plot_df = pivot_df.melt(id_vars='time_bin', var_name='team', value_name='probability')

    # Create visualization
    plt.figure(figsize=(12, 6))
    for team in teams:
        team_df = plot_df[plot_df['team'] == team]
        plt.plot(team_df['time_bin'], team_df['probability'], 
                label=team, marker='o', linestyle='-')

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
    # Example usage:
    input_csv = "der_klassiker.csv"  # Replace with your CSV path
    output_csv = "analyzed_results.csv"
    
    # Analyze sentiments
    analyzed_df = analyze_sentiments(input_csv, output_csv)
    
    # Visualize results
    visualize_sentiments(output_csv)
