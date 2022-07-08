import os
import json
import torch
import requests
import numpy as np
import pandas as pd
import lyricsgenius as genius
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer

abs = os.path.abspath(__file__)
dir = os.path.dirname(abs)
os.chdir(dir)

#----------------------------------------------------------------#

client_id = open('.client_id', 'r')
client_secret = open('.client_secret', 'r')

auth_url = 'https://accounts.spotify.com/api/token'

auth_req = requests.post(auth_url, {'grant_type': 'client_credentials',
                                    'client_id': client_id,
                                    'client_secret': client_secret})

auth_resp = auth_req.json()
sp_token = auth_resp['access_token']

headers = {'Authorization': 'Bearer {token}'.format(token=sp_token)}

album_get = requests.get('https://api.spotify.com/v1/albums/6NTrwu1XJ56jBPx7HMksbZ/tracks',
                         headers=headers)

#----------------------------------------------------------------#

tracks_get = album_get.json()['items']

trkfeat = []

for track in tracks_get:
    features = requests.get('https://api.spotify.com/v1/audio-features/' + track['id'],
                            headers=headers)
    features = features.json()
    features.update({'track_name': track['name']})
    trkfeat.append(features)

albumfeat = pd.DataFrame(trkfeat)
albumfeat = albumfeat[['track_name', 'danceability', 'energy', 'speechiness']]

albumfeat['track_name'] = albumfeat['track_name'].str.replace('- 2015 Remaster', '', regex=True)

#----------------------------------------------------------------#

gns = genius.Genius(open('.genius_token', 'r'),
                    remove_section_headers=True,
                    skip_non_songs=True,
                    verbose=False,
                    excluded_terms = ['(Remix)', '(Live)', '(Mix)', '(Edit)',
                                      '(Version)', '(Extended)', '(Remaster)',
                                      '(Demo)', '(Writing Session)', '(Outtake)'])

get_lyrics = gns.search_album('Power, Corruption & Lies', 'New Order')

get_lyrics.save_lyrics()

#----------------------------------------------------------------#

lyrics_get = json.load(open('Lyrics_PowerCorruptionLies.json'))

lyrics_get = lyrics_get.get('tracks')
lyrics_get = pd.json_normalize(lyrics_get)

lyrics_get = lyrics_get[['song.title', 'song.lyrics']]

lyrics_get['song.title'] = lyrics_get['song.title'].str.replace('by New Order', '', regex=True)

lyrics_get['song.lyrics'] = lyrics_get['song.lyrics'].str.replace('^.*(?:...)Lyrics', '', regex=True)
lyrics_get['song.lyrics'] = lyrics_get['song.lyrics'].str.replace('\d+' + 'Embed', '', regex=True)
lyrics_get['song.lyrics'] = lyrics_get['song.lyrics'].str.replace('Embed', '', regex=True)
lyrics_get['song.lyrics'] = lyrics_get['song.lyrics'].str.replace('\n', ' ', regex=True)

album = albumfeat.join(lyrics_get).rename(columns={'track_name': 'Title',
                                                   'danceability': 'Danceability',
                                                   'energy' : 'Energy',
                                                   'speechiness': 'Speechiness',
                                                   'song.lyrics': 'Lyrics'})

album.drop('song.title', axis=1, inplace=True)

#----------------------------------------------------------------#

class DataSet:
    def __init__(self, token_txt):
        self.token_txt = token_txt
    
    def __len__(self):
        return len(self.token_txt['input_ids'])
    
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.token_txt.items()}

#----------------------------------------------------------------#

model_get = 'j-hartmann/emotion-english-distilroberta-base'
tokenizer = AutoTokenizer.from_pretrained(model_get)
model = AutoModelForSequenceClassification.from_pretrained(model_get)
trainer = Trainer(model=model)

lyrics = album['Lyrics'].tolist()

token_txt = tokenizer(lyrics, 
                      truncation=True, 
                      padding=True)

txt_inp = DataSet(token_txt)

sentiment = trainer.predict(txt_inp)

#----------------------------------------------------------------#

pred = sentiment.predictions.argmax(-1)
label = pd.Series(pred).map(model.config.id2label)
raw_score = (np.exp(sentiment[0]) / np.exp(sentiment[0]).sum(-1, keepdims=True))

anger = []
disgust = []
fear = []
joy = []
neutral = []
sadness = []
surprise = []

for i in range(len(lyrics)):
    anger.append(raw_score[i][0])
    disgust.append(raw_score[i][1])
    fear.append(raw_score[i][2])
    joy.append(raw_score[i][3])
    neutral.append(raw_score[i][4])
    sadness.append(raw_score[i][5])
    surprise.append(raw_score[i][6])

result = pd.DataFrame(list(zip(label, anger, disgust, fear, joy,
                               neutral, sadness, surprise, lyrics)),
                               columns=['Sentiment', 'Anger',
                                        'Disgust', 'Fear', 'Joy',
                                        'Neutral', 'Sadness',
                                        'Surprise', 'Lyrics'])

result['Sentiment'] = result['Sentiment'].str.capitalize()

#----------------------------------------------------------------#

final_album = album.merge(result, on='Lyrics')

shift1 = final_album.pop('Lyrics')
shift2 = final_album.pop('Sentiment')

final_album.insert(11, 'Lyrics', shift1)
final_album.insert(1, 'Sentiment', shift2)

final_album.to_csv('NewOrder_PCL_Sentiment.csv', index=False)
