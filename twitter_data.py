import csv
import requests
import pandas as pd
import html
from config import API_KEY, API_SECRET, BEARER_TOKEN

# Set up your API credentials
api_key = API_KEY
api_secret = API_SECRET
bearer_token = BEARER_TOKEN


# Set up the API endpoint and query parameters
endpoint = "https://api.twitter.com/2/tweets/search/recent"
count = 100  # Number of tweets to retrieve

# Set up the HTTP headers
headers = {
    "Authorization": f"Bearer {bearer_token}",
}


def collectData(keyword):
    # Set up the HTTP parameters
    params = {
        "query": f'{keyword} lang:en -is:retweet',
        "max_results": count,
    }

    # Make the API request
    response = requests.get(endpoint, headers=headers, params=params)
    requiredTweets = []

    # Check the status code of the response
    if response.status_code == 200:
        try:
            response_json = response.json()
            tweets = response.json()["data"]
            for tweet in tweets:
                requiredTweets.append(html.unescape(tweet['text']))
            df = pd.DataFrame(requiredTweets)
            df.to_csv('csv_file.csv', mode='a', header=['text'], index=False)
        except KeyError:
            print("No Data Found For the Given Query.")
    else:
        print(response.json())