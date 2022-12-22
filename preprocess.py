import os
import json
import requests
import numpy as np
import pandas as pd
import lyricsgenius as genius
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer

abs = os.path.abspath(__file__)
dir = os.path.dirname(abs)
os.chdir(dir)

#----------------------------------------------------------------#

sp_client_id = open('.client_id', 'r')
sp_client_secret = open('.client_secret', 'r')

auth_url = 'https://accounts.spotify.com/api/token'

auth_req = requests.post(auth_url, {'grant_type': 'client_credentials',
                                    'client_id': sp_client_id,
                                    'client_secret': sp_client_secret})

auth_resp = auth_req.json()
sp_token = auth_resp['access_token']

headers = {'Authorization': 'Bearer {token}'.format(token=sp_token)}

get_album = requests.get('https://api.spotify.com/v1/albums/6NTrwu1XJ56jBPx7HMksbZ/tracks',
                         headers=headers)

#----------------------------------------------------------------#

get_tracks = get_album.json()['items']

track_feat = []

for track in get_tracks:
    features = requests.get('https://api.spotify.com/v1/audio-features/' + track['id'],
                            headers=headers)
    features = features.json()
    features.update({'track_name': track['name']})
    track_feat.append(features)

album_feat = pd.DataFrame(track_feat)
album_feat = album_feat[['track_name', 'danceability', 'energy', 'speechiness']]

album_feat['track_name'] = album_feat['track_name'].str.replace('- 2015 Remaster', '', regex=True)

#----------------------------------------------------------------#

gns = genius.Genius(open('.genius_token').read(),
                    remove_section_headers=True,
                    skip_non_songs=True,
                    verbose=False,
                    excluded_terms = ['(Remix)', '(Live)', '(Mix)', '(Edit)',
                                      '(Version)', '(Extended)', '(Remaster)',
                                      '(Demo)', '(Writing Session)', '(Outtake)'])

get_lyrics = gns.search_album('Power, Corruption & Lies', 'New Order')

get_lyrics.save_lyrics()

#----------------------------------------------------------------#

raw_lyrics = json.load(open('Lyrics_PowerCorruptionLies.json'))

raw_lyrics = raw_lyrics.get('tracks')
raw_lyrics = pd.json_normalize(raw_lyrics)

raw_lyrics = raw_lyrics[['song.title', 'song.lyrics']]

def clean_lyrics(text):
    text = text.str.replace('by New Order', '',)
    text = text.str.replace('You might also like', '')
    text = text.str.replace('^.*(?:...)Lyrics', '', regex=True)
    text = text.str.replace('\d+' + 'Embed', '', regex=True)
    text = text.str.replace('Embed', '', regex=True)
    text = text.str.replace('\n', ' ', regex=True)
    text = text.str.strip()
    return text

scrubbed_lyrics = raw_lyrics.apply(clean_lyrics)

album = album_feat.join(scrubbed_lyrics).rename(columns={'track_name': 'Title',
                                                         'danceability': 'Danceability',
                                                         'energy' : 'Energy',
                                                         'speechiness': 'Speechiness',
                                                         'song.lyrics': 'Lyrics'})

album = album.drop('song.title', axis=1)

#----------------------------------------------------------------#

class DataSet:
    def __init__(self, token_txt):
        self.token_txt = token_txt
    
    def __len__(self):
        return len(self.token_txt['input_ids'])
    
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.token_txt.items()}

#----------------------------------------------------------------#

get_model = 'j-hartmann/emotion-english-distilroberta-base'
tokenizer = AutoTokenizer.from_pretrained(get_model)
model = AutoModelForSequenceClassification.from_pretrained(get_model)
trainer = Trainer(model=model)

lyrics = album['Lyrics'].tolist()

token_lyrics = tokenizer(lyrics, 
                         truncation=True, 
                         padding=True)

input_lyrics = DataSet(token_lyrics)

sentiment = trainer.predict(input_lyrics)

#----------------------------------------------------------------#

pred = sentiment.predictions.argmax(-1)
label = pd.Series(pred).map(model.config.id2label)
score = (np.exp(sentiment[0]) / np.exp(sentiment[0]).sum(-1, keepdims=True))

anger = []
disgust = []
fear = []
joy = []
neutral = []
sadness = []
surprise = []

for i in range(len(lyrics)):
    anger.append(score[i][0])
    disgust.append(score[i][1])
    fear.append(score[i][2])
    joy.append(score[i][3])
    neutral.append(score[i][4])
    sadness.append(score[i][5])
    surprise.append(score[i][6])

model_result = pd.DataFrame(list(zip(label, anger, disgust, fear, joy,
                                     neutral, sadness, surprise, lyrics)),
                            columns=['Sentiment', 'Anger',
                                     'Disgust', 'Fear', 'Joy',
                                     'Neutral', 'Sadness',
                                     'Surprise', 'Lyrics'])

model_result['Sentiment'] = model_result['Sentiment'].str.capitalize()

#----------------------------------------------------------------#

final_result = album.merge(model_result, on='Lyrics')

shift1 = final_result.pop('Lyrics')
shift2 = final_result.pop('Sentiment')

final_result.insert(11, 'Lyrics', shift1)
final_result.insert(1, 'Sentiment', shift2)

final_result.to_csv('NewOrder_PCL_Sentiment.csv', index=False)
