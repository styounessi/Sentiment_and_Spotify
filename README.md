# Album Sentiment & Spotify Audio Features
Measuring the emotional qualities of an album using Spotify Track Audio Features with lyrical sentiment analysis via RoBERTa. The data is gathered,
processed, and merged in a .py file and then used in a Jupyter Notebook to build a summary presentation. 

## Requirements
You can install any of the required libraries using `pip` like in the example below.

`pip install pandas`

- [Plotly](https://pypi.org/project/plotly/)
- [Pandas](https://pypi.org/project/pandas/)
- [Torch](https://pypi.org/project/torch/)
- [Requests](https://pypi.org/project/requests/)
- [NumPy](https://pypi.org/project/numpy/)
- [LyricsGenius](https://pypi.org/project/lyricsgenius/)

You will also need access to the [Spotify](https://developer.spotify.com/documentation/web-api/quick-start/) and [Genius](https://docs.genius.com/) APIs. 

Any suitable model can be used from [🤗 Hugging Face](https://huggingface.co/) with some modifications but `j-hartmann/emotion-english-distilroberta-base` is used in this case.

## Power, Corruption, & Lies by New Order
![PCL by New Order album art](https://cps-static.rovicorp.com/3/JPG_500/MI0003/239/MI0003239337.jpg)

The album used is [*Power, Corruption & Lies*](https://en.wikipedia.org/wiki/Power,_Corruption_%26_Lies) by [New Order](https://en.wikipedia.org/wiki/New_Order_(band)). The original track list is used with these eight tracks:
1. Age of Consent 
2. We All Stand
3. The Village
4. 5 8 6
5. Your Silent Face
6. Ultraviolence
7. Ecstasy
8. Leave Me Alone
