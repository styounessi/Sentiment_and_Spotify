# Album Sentiment & Spotify Audio Features
Measuring the qualities of an album using Spotify Track Audio Features plus lyrical sentiment analysis via RoBERTa. The data is gathered,
pre-processed, run through the model, and merged in a .py file while the Jupyter Notebook contains a presentable summary with visualizations.

## Requirements
You can install any of the required libraries using `pip`.

`pip install pandas`

- [Plotly](https://pypi.org/project/plotly/)
- [Pandas](https://pypi.org/project/pandas/)
- [Torch](https://pypi.org/project/torch/)
- [Requests](https://pypi.org/project/requests/)
- [NumPy](https://pypi.org/project/numpy/)
- [LyricsGenius](https://pypi.org/project/lyricsgenius/)

You will need access to the [Spotify](https://developer.spotify.com/documentation/web-api/quick-start/) and [Genius](https://docs.genius.com/) APIs. 

Any model can be used from [🤗 Hugging Face](https://huggingface.co/) with some modifications but `j-hartmann/emotion-english-distilroberta-base` is used in
this case.

## Power, Corruption, & Lies by New Order
![PCL by New Order album art](https://cps-static.rovicorp.com/3/JPG_500/MI0003/239/MI0003239337.jpg)

The album used is *Power, Corruption & Lies* by New Order. The original track list is used with these eight tracks:
1. Age of Consent 
2. We All Stand
3. The Village
4. 5 8 6
5. Your Silent Face
6. Ultraviolence
7. Ecstasy
8. Leave Me Alone
