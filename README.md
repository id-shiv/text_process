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

### DATABASE

DB_HOST_NAME = '127.0.0.1'  
DB_PORT = 9201  

### ENCODER

ENCODER_PATH = '/Users/shiv/Documents/gitRepositories/text_search/encoders/universal-sentence-encoder-large_5'   
_encoder = hub.load(ENCODER_PATH)  # Load the encoder

## Results

`NOTE: Results depend on # of samples in dataset, current project could be improved with some pre-processing of text`

Search Text = 'psychological thriller is what i like'

Results:  
{'percentage_match': 73, 'text': "Well, I like to watch bad horror B-Movies, cause I think it's interesting to see stupidity and unability of creators to shoot seriously good movie"}  
{'percentage_match': 70, 'text': '"The Cell" is an exotic masterpiece, a dizzying trip into not only the vast mind of a serial killer, but also into one of a very talented director'}  
{'percentage_match': 67, 'text': "Average (and surprisingly tame) Fulci giallo which means it's still quite bad by normal standards, but redeemed by its solid build-up and some nice touches such as a neat time twist on the issues of visions and clairvoyance"}  
{'percentage_match': 67, 'text': 'Taut and organically gripping, Edward Dmytryk\'s Crossfire is a distinctive suspense thriller, an unlikely "message" movie using the look and devices of the noir cycle'}  
{'percentage_match': 65, 'text': 'This movie struck home for me'}  
{'percentage_match': 65, 'text': 'How this film could be classified as Drama, I have no idea'}  
{'percentage_match': 64, 'text': 'This film took me by surprise'}  
{'percentage_match': 64, 'text': 'This film laboured along with some of the most predictable story lines and shallow characters ever seen'}  
{'percentage_match': 63, 'text': 'Oh noes one of these attack of the Japanese ghost girl movies'}  
{'percentage_match': 63, 'text': 'I really like Salman Kahn so I was really disappointed when I seen this movie'}  
# text-cluster
