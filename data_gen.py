import pandas as pd
from datetime import datetime, timedelta
import random

def generate_sample_data(match_start, teams, num_tweets):
    tweets = []
    for _ in range(num_tweets):
        team = random.choice(teams)
        sentiment = random.choice(['positive', 'negative', 'neutral'])

        # Generate realistic football action tweets
        if sentiment == 'positive':
            content = random.choice([
                f"What a goal from {team}! 🔥 #{team}",
                f"Brilliant save by {team}'s goalkeeper! 🛡 #{team}",
                f"Clinical finish by {team}'s striker! ⚽ #{team}",
                f"Amazing tackle by {team}'s defender! 💪 #{team}",
                f"Perfect through ball from {team}! 🎯 #{team}"
            ])
        elif sentiment == 'negative':
            content = random.choice([
                f"Horrible miss by {team}! 😡 #{team}",
                f"Clear foul by {team}'s midfielder! 🟥 #{team}",
                f"Poor penalty from {team}! 🙅♂️ #{team}",
                f"Awful defensive error by {team}! 🤦♂️ #{team}",
                f"Wasted chance by {team}'s forward! 😫 #{team}"
            ])
        else:
            content = random.choice([
                f"Decent tackle by {team} ⚔ #Football",
                f"{team} making tactical substitutions ↔ #{team}",
                f"Controversial offside call against {team} ⚖ #{team}",
                f"Good interception by {team}'s defender 🛑 #{team}"
            ])

        # Generate timestamp within match duration
        tweet_time = match_start + timedelta(
            minutes=random.randint(0, 90),
            seconds=random.randint(0, 59)
        )

        tweets.append({
            'timestamp': tweet_time.strftime('%Y-%m-%d %H:%M:%S'),
            'tweet': content,
            'team': team
        })

    return pd.DataFrame(tweets)

# Match details and dataset generation
matches = [
    {'teams': ['Manchester City', 'Manchester United'], 'file': 'manchester_derby.csv'},
    {'teams': ['Liverpool', 'PSG'], 'file': 'liverpool_psg.csv'},
    {'teams': ['Dortmund', 'Bayern Munchen'], 'file': 'der_klassiker.csv'}
]

match_start = datetime(2025, 4, 4, 19, 0)  # Match start time

for match in matches:
    df = generate_sample_data(match_start, match['teams'], num_tweets=50)
    df.to_csv(match['file'], index=False)
