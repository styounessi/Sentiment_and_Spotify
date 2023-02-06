import json
import requests
import numpy as np
import pandas as pd
import lyricsgenius as genius
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer

#-----------------------Get Album via Spotify API-----------------------#

# Read Spotify client_id and client_secret keys
with open('.client_id', 'r') as f:
    sp_client_id = f.read().strip()

with open('.client_secret', 'r') as f:
    sp_client_secret = f.read().strip()

# Connect to Spotify and post request
auth_url = 'https://accounts.spotify.com/api/token'
auth_req = requests.post(auth_url, {'grant_type': 'client_credentials',
                                    'client_id': sp_client_id,
                                    'client_secret': sp_client_secret})

auth_resp = auth_req.json()
sp_token = auth_resp['access_token']

headers = {'Authorization': f'Bearer {sp_token}'}

# Get specific album with tracks
get_album = requests.get('https://api.spotify.com/v1/albums/6NTrwu1XJ56jBPx7HMksbZ/tracks',
                         headers=headers)

#-----------------------Get Spotify Audio Features-----------------------#

get_tracks = get_album.json()['items']

# Grab track titles and rename to remove unnecessary text
track_names = [track['name'] for track in get_tracks]
track_names = [name.replace('- 2015 Remaster', '') for name in track_names]

track_ids = [track['id'] for track in get_tracks]
track_feat = [requests.get(f'https://api.spotify.com/v1/audio-features/{id}',
                           headers=headers).json() for id in track_ids]

for track, name in zip(track_feat, track_names):
    track.update({'track_name': name})

# Convert to DataFrame and only keep a few audio features to append with later
album_feat = pd.DataFrame(track_feat)
album_feat = album_feat[['track_name', 'danceability', 'energy', 'speechiness']]

#-----------------------Get Lyrics from Genius API-----------------------#

# Use Genius API key with LyricsGenius, remove unnecessary terms
gns = genius.Genius(open('.genius_token').read(),
                    remove_section_headers=True,
                    skip_non_songs=True,
                    verbose=False,
                    excluded_terms = ['(Remix)', '(Live)', '(Mix)', '(Edit)',
                                      '(Version)', '(Extended)', '(Remaster)',
                                      '(Demo)', '(Writing Session)', '(Outtake)'])


# Search for specific album, save scraped results as JSON
get_lyrics = gns.search_album('Power, Corruption & Lies', 'New Order')
get_lyrics.save_lyrics()

#-----------------------Scrub Lyrics for Tokenizer-----------------------#

# Load lyrics JSON
with open('Lyrics_PowerCorruptionLies.json', 'r') as f:
    raw_lyrics = json.load(f)

# Specifically seeking lyrics from each album track, initialized as pandas DataFrame
raw_lyrics = raw_lyrics.get('tracks')
raw_lyrics = pd.json_normalize(raw_lyrics)
raw_lyrics = raw_lyrics[['song.title', 'song.lyrics']]

def clean_lyrics(text):
    '''
    Scrub unnecessary strings from the scraped lyrics.
    
    Args:
        text (str): The text of scraped lyrics.
    
    Returns:
        str: The cleaned text of the scraped lyrics.
    '''
    text = text.str.replace('by New Order', '',)
    text = text.str.replace('You might also like', '')
    text = text.str.replace('See New Order LiveGet tickets as low as \$\d+', '', regex=True)
    text = text.str.replace('^.*(?:...)Lyrics', '', regex=True)
    text = text.str.replace('\d+' + 'Embed', '', regex=True)
    text = text.str.replace('Embed', '')
    text = text.str.replace('\n', ' ')
    text = text.str.strip()
    return text

scrubbed_lyrics = raw_lyrics.apply(clean_lyrics)

# Joining scrubbed lyrics with previous album features DataFrame
album = album_feat.join(scrubbed_lyrics).rename(columns={'track_name': 'Title',
                                                         'danceability': 'Danceability',
                                                         'energy' : 'Energy',
                                                         'speechiness': 'Speechiness',
                                                         'song.lyrics': 'Lyrics'})

# This column is now redundant given the 'Title' column is there after append
album = album.drop('song.title', axis=1)

#-----------------------Define Tokenized DataSet Class-----------------------#

class DataSet:
    '''
    Represents a dataset of tokenized text data.
    
    Attributes:
        token_txt (dict): A dictionary containing tokenized text data. 
    '''
    
    def __init__(self, token_txt):
        '''
        Initialize the DataSet object with the tokenized text data.

        Args:
            token_txt (dict): A dictionary containing the tokenized text data.
        '''
        self.token_txt = token_txt
    
    def __len__(self):
        '''
        Return the length of the input_ids component of the token_txt data.

        Returns:
            int: The length of the input_ids component of the token_txt data.
        '''
        return len(self.token_txt['input_ids'])
    
    def __getitem__(self, idx):
        '''
        Retrieve the values at the given index.

        Args:
            idx (int): The index of the values to retrieve.

        Returns:
            dict: A dictionary containing the values at the given index.
        '''
        return {k: v[idx] for k, v in self.token_txt.items()}

#-----------------------Get & Run Sentiment Model-----------------------#

# Get sentiment model and initialize tokenizer, model, and trainer
get_model = 'j-hartmann/emotion-english-distilroberta-base'
tokenizer = AutoTokenizer.from_pretrained(get_model)
model = AutoModelForSequenceClassification.from_pretrained(get_model)
trainer = Trainer(model=model)

# Convert lyrics to list and tokenize
lyrics = album['Lyrics'].tolist()
token_lyrics = tokenizer(lyrics, truncation=True, padding=True)

# Derive sentiment predictions via trainer
input_lyrics = DataSet(token_lyrics)
sentiment = trainer.predict(input_lyrics)

#-----------------------Append Sentiment Scores-----------------------#

# Prepare predictions for final append
pred = sentiment.predictions.argmax(-1)
label = pd.Series(pred).map(model.config.id2label)
scores = (np.exp(sentiment[0]) / np.exp(sentiment[0]).sum(-1, keepdims=True))

anger = []
disgust = []
fear = []
joy = []
neutral = []
sadness = []
surprise = []

# Appends the corresponding emotional score from list of scores to lyrics list
for i, lyric in enumerate(lyrics):
       anger.append(scores[i][0])
       disgust.append(scores[i][1])
       fear.append(scores[i][2])
       joy.append(scores[i][3])
       neutral.append(scores[i][4])
       sadness.append(scores[i][5])
       surprise.append(scores[i][6])

# Convert results to DataFrame and rename column headers
model_result = pd.DataFrame(list(zip(label, anger, disgust, fear, joy,
                                     neutral, sadness, surprise, lyrics)),
                            columns=['Sentiment', 'Anger',
                                     'Disgust', 'Fear', 'Joy',
                                     'Neutral', 'Sadness',
                                     'Surprise', 'Lyrics'])

model_result['Sentiment'] = model_result['Sentiment'].str.capitalize()

#-----------------------Finalize Resuls-----------------------#

# Final merge between sentiment results DataFrame and previous album DataFrame
final_result = album.merge(model_result, on='Lyrics')

# Shifting the 'Lyrics' and 'Sentiment' columns to better positions for readability
shift1 = final_result.pop('Lyrics')
shift2 = final_result.pop('Sentiment')
final_result.insert(11, 'Lyrics', shift1)
final_result.insert(1, 'Sentiment', shift2)

# Save to CSV
final_result.to_csv('NewOrder_PCL_Sentiment.csv', index=False)
