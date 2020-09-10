# text_tag

## Problem Statement  

- For a given text, identify most relevant tags.

## Approach

- For this project, let's use IMDB dataset of movie reviews and generate optimal set of clusters.
- To generate optimal set of clusters, we shall use `sum of squared distances` and plot to view the K at elbow.

## Requirements

- Universal Sentence Encoder (USE) model downloaded and stored locally from <https://tfhub.dev/google/universal-sentence-encoder-large/5>
- IMDB Dataset downloaded and stored locally from <https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews>
- Installs:  
`pip3 install --upgrade pip`  
`pip3 install pandas`  
`pip3 install nltk`  
`pip3 install sklearn`  
`pip3 install matplotlib`  

## Configurations

### DATA

DATA_PATH = '/Users/shiv/Documents/gitRepositories/iutils/input/data/IMDB Dataset.csv'  
TEXT_COLUMN = 'review'  
NUM_OF_SAMPLES = 100  

### ENCODER

ENCODER_PATH = '/Users/shiv/Documents/gitRepositories/text_search/encoders/universal-sentence-encoder-large_5'   
_encoder = hub.load(ENCODER_PATH)  # Load the encoder

## Results

`NOTE: Results depend on # of samples in dataset, current project could be improved with some pre-processing of text`

Search Text = 'psychological thriller is what i like'
