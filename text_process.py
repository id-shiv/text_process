# region Import packages
from tqdm import tqdm

import re
import string
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer, PorterStemmer
# nltk.download('stopwords')
# nltk.download('wordnet')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

from collections import Counter

#endregion


def _remove_stopwords(text: str, stop_words: list()):
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in stop_words])

def _get_words_frequenct(df: pd.DataFrame(), text_column: str, n_frequent: int=10):
    """custom function to remove the frequent words"""
    cnt = Counter()
    for text in df[text_column].values:
        for word in text.split():
            cnt[word] += 1
            
    freq_words = set([w for (w, wc) in cnt.most_common(n_frequent)])
    rare_words = set([w for (w, wc) in cnt.most_common()[:-n_frequent-1:-1]])

    return freq_words, rare_words

def pre_process(df: pd.DataFrame(), text_column: str,
                text_lower: bool=True,
                remove_html_tags: bool=True,
                remove_punctuations: bool=True, punctuations: list()=string.punctuation, remove_digits: bool=True, 
                remove_stopwords: bool=True, stop_words: list()=stopwords.words('english'), additional_stopwords: list()=None,
                remove_frequent_words: bool=True, n_frequent: int=10,
                remove_rare_words: bool=True,
                stem_words: bool=False, lemmatize_words: bool=True):
    '''
    Takes in a dataframe and a text column name, then processes these texts.

    ::RETURN
    Input dataframe with an added column 'pre_processed' containing processed texts of 'text_column'

    # Reference: https://www.kaggle.com/sudalairajkumar/getting-started-with-text-preprocessing
    '''
    
    df['processed_text'] = np.nan  # create a new column to contain the processed texts
    stemmer = PorterStemmer()  # create the stemmer
    lemmatizer = WordNetLemmatizer()  # create the lemmatizer

    # get n most frequently occuring and rare words in entire corpus
    _freq_words, _rare_words = _get_words_frequenct(df, text_column, n_frequent)

    print('Processing texts')
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        text = row[text_column]
        
        if text_lower:
            text = text.lower()
        
        if remove_html_tags:
            html_pattern = re.compile('<.*?>')
            text = html_pattern.sub(r'', text)

        if remove_punctuations:
            # map punctuation to space
            translator = str.maketrans(punctuations, ' '*len(punctuations)) 
            text = text.translate(translator)

        if remove_digits:
            text = re.sub('\d', ' ', text)

        if remove_stopwords:
            if additional_stopwords:
                stop_words = stop_words + additional_stopwords
            text = _remove_stopwords(text, stop_words)
        
        if remove_frequent_words:
            text = " ".join([word for word in str(text).split() if word not in _freq_words])

        if remove_rare_words:
            text = " ".join([word for word in str(text).split() if word not in _rare_words])

        if stem_words:
            text = " ".join([stemmer.stem(word) for word in text.split()])

        if lemmatize_words:
            text = " ".join([lemmatizer.lemmatize(word, "n") for word in text.split()])

        # print(row[text_column])
        # print('\n\n-----\n\n')
        # print(text)
        df.iloc[index, df.columns.get_loc('processed_text')] = text

    return df

if __name__ == "__main__":
    #region CONSTANTS

    ## DATA
    DATA_PATH = '/Users/shiv/Documents/gitRepositories/iutils/input/data/IMDB Dataset.csv'
    TEXT_COLUMN = 'review'
    NUM_OF_SAMPLES = 2

    #endregion

    df = pd.read_csv(DATA_PATH)
    df_preprocessed = pre_process(df, text_column=TEXT_COLUMN)
    # print(df_preprocessed)