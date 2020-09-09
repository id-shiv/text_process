# region Import packages
from tqdm import tqdm

import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
# nltk.download('stopwords')
# nltk.download('wordnet')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

#endregion

#region CONSTANTS

## DATA
DATA_PATH = '/Users/shiv/Documents/gitRepositories/iutils/input/data/IMDB Dataset.csv'
TEXT_COLUMN = 'review'
NUM_OF_SAMPLES = 5000

#endregion

def get_texts(data_path: str, text_column: str, num_of_samples: int=-1):
    # Read the dataset and retrieve texts
    data = pd.read_csv(data_path)

    # Let's use only the first sentence of the text for our project
    data = data[text_column].apply(lambda x: x.split('.')[0])

    if num_of_samples <= 0:
        return data
    else:
        return data[:num_of_samples]

def tokenize(sentences):
    tokenizer = RegexpTokenizer(r'\w+')
    sample_data_tokenized = [w.lower() for w in sentences]
    sample_data_tokenized = [tokenizer.tokenize(i) for i in sample_data_tokenized]
    
    return(sample_data_tokenized)

def content_extractor(content: str, start: str=None, end: str=None):
    try:
        if start and content and end:
            builder = "{}(.*)(?={})".format(start, end)
            pattern = re.compile(builder)
            return pattern.search(content).group(0)
        else:
            return content
    except Exception as e:
            return content
    
    # USAGE:
    # ser1=df.apply(lambda x: content_extractor(x,"start_text","end_text"))

def text_process(text: str):
    '''
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Return the cleaned text as a list of words
    4. Remove words
    '''
    stemmer = WordNetLemmatizer()
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join([i for i in nopunc if not i.isdigit()])
    nopunc =  [word.lower() for word in nopunc.split() if word not in stopwords.words('english')]
    return [stemmer.lemmatize(word) for word in nopunc]

    # USAGE:
    # sample_text = "Hey There! This is a Sample review, which 123happens {blah}%456 to contain happened punctuations universal rights of right contained."
    # print(text_process(sample_text))

def vectorize(texts: list()):
    """ TFIDF Vectorizer is used to create a vocabulary. 
    TFIDF is a product of how frequent a word is in a document multiplied by how unique a word is w.r.t the entire corpus. 
    ngram_range parameter : which will help to create one , two or more word vocabulary depending on the requirement.
    """
    vectorizer = TfidfVectorizer()
    X_vectorized = vectorizer.fit_transform(texts)

    return X_vectorized, vectorizer

def optimal_k(X_vectorized, k_max=5):
    # Determine Optimal K value
    sum_of_sq_distances = list()
    K_range = range(1, k_max)
    
    for i in tqdm(K_range):
        model = KMeans(n_clusters=i)    
        model.fit(X_vectorized)
        sum_of_sq_distances.append(model.inertia_)

    plt.plot(K_range, sum_of_sq_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum of Squared Distances')
    plt.title('Find Optimal K using Elbow method')
    plt.show()
    # K at the elbow of the curve in above graph would be an optimal K

if __name__ ==  '__main__':
    texts = get_texts(DATA_PATH, TEXT_COLUMN, NUM_OF_SAMPLES)
    
    print('Processing texts...')
    texts = [' '.join(text_process(text)) for text in tqdm(texts)]
    X_vectorized, vectorizer = vectorize(texts)

    # Find Optimal K when required
    # print('Determining optimal K using Sum of Squared Distances...')
    # optimal_k(X_vectorized, k_max=10)

    K = 6
    model = KMeans(n_clusters=K, init='k-means++', max_iter=100, n_init=1)
    model.fit(X_vectorized)

    print("Top terms per cluster:")
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    for i in range(K):
        print("Cluster %d:" % i),
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind]),
        print

    # Prediction
    print("\n")
    print("Prediction")

    Y = vectorizer.transform(["this movie is a horror thriller"])
    prediction = model.predict(Y)
    print(prediction)

    Y = vectorizer.transform(["drama movies are often too long."])
    prediction = model.predict(Y)
    print(prediction)
    
