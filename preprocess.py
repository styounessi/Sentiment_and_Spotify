import os
import json
import requests
import pandas as pd
import lyricsgenius as genius

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

album = requests.get('https://api.spotify.com/v1/albums/6NTrwu1XJ56jBPx7HMksbZ/tracks',
                      headers=headers)

#----------------------------------------------------------------#

tracks = album.json()['items']

trkfeat = []

for track in tracks:
    features = requests.get('https://api.spotify.com/v1/audio-features/' + track['id'],
                             headers=headers)
    features = features.json()
    features.update({'track_name': track['name'],})
    trkfeat.append(features)

albumfeat = pd.DataFrame(trkfeat)
albumfeat = albumfeat[['track_name', 'danceability', 'energy', 'speechiness']]

albumfeat['track_name'] = albumfeat['track_name'].str.replace('- 2015 Remaster', '', regex=True)

#----------------------------------------------------------------#

gns = genius.Genius(open('.genius_token').read(),
                    remove_section_headers = True,
                    skip_non_songs = True,
                    verbose = False,
                    excluded_terms = ['(Remix)', '(Live)', '(Mix)', '(Edit)',
                                      '(Version)', '(Extended)', '(Remaster)',
                                      '(Demo)', '(Writing Session)', '(Outtake)'])

get_lyrics = gns.search_album('Power, Corruption & Lies', 'New Order')

get_lyrics.save_lyrics()

#----------------------------------------------------------------#

lyrics = json.load(open('Lyrics_PowerCorruptionLies.json'))

lyrics = lyrics.get('tracks')
lyrics = pd.json_normalize(lyrics)

lyrics = lyrics[['song.title', 'song.lyrics']]

lyrics['song.title'] = lyrics['song.title'].str.replace('by New Order', '', regex=True)

lyrics['song.lyrics'] = lyrics['song.lyrics'].str.replace('^.*(?:...)Lyrics', '', regex=True)
lyrics['song.lyrics'] = lyrics['song.lyrics'].str.replace('\d+' + 'Embed', '', regex=True)
lyrics['song.lyrics'] = lyrics['song.lyrics'].str.replace('Embed', '', regex=True)
lyrics['song.lyrics'] = lyrics['song.lyrics'].str.replace('\n', ' ', regex=True)

final_album = albumfeat.join(lyrics).rename(columns={'track_name': 'Title',
                                                     'danceability': 'Danceability',
                                                     'energy' : 'Energy',
                                                     'speechiness': 'Speechiness',
                                                     'song.lyrics': 'Lyrics'})

final_album.drop('song.title', axis=1, inplace=True)

final_album.to_csv('New_Order_PCL.csv')