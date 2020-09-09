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

def pre_process(text: str):
    '''
    Takes in a string of text, then performs the following:
    1. Remove all punctuation and digits
    2. Remove all stopwords
    3. Return the cleaned text as a list of words
    4. Remove words
    '''
    stemmer = WordNetLemmatizer()
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join([i for i in nopunc if not i.isdigit()])
    nopunc =  [word.lower() for word in nopunc.split() if word not in stopwords.words('english')]
    return " ".join([stemmer.lemmatize(word) for word in nopunc])

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
    # Read the dataset and retrieve texts
    data = pd.read_csv(DATA_PATH)

    # Let's use only the first sentence of the text for our project
    data[TEXT_COLUMN] = data[TEXT_COLUMN].apply(lambda x: x.split('.')[0])

    if NUM_OF_SAMPLES > 0:
        data = data.head(NUM_OF_SAMPLES)
    
    print('Pre-Processing texts...')
    data[TEXT_COLUMN] = data[TEXT_COLUMN].apply(lambda x: pre_process(x))
    # print(data[TEXT_COLUMN].tolist()[:5])
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.cluster import KMeans
    import numpy as np
    import pandas as pd

    # In information retrieval or text mining, the term frequency-inverse document frequency also called tf-idf, 
    # is a well known method to evaluate how important is a word in a document. 
    # tf-idf are also a very interesting way to convert the textual representation of information into a Vector Space Model (VSM).
    vectorizer_tfidf = TfidfVectorizer(stop_words='english')
    X = vectorizer_tfidf.fit_transform(data[TEXT_COLUMN].tolist())
    
    # It takes the words of each sentence and creates a vocabulary of all the unique words in the sentences. 
    # This vocabulary can then be used to create a feature vector of the count of the words:
    vectorizer_count = CountVectorizer(min_df=0, lowercase=False)
    vectorizer_count.fit(data[TEXT_COLUMN].tolist())
    print(list(vectorizer_count.vocabulary_)[:5])

    # # Find Optimal K when required
    # print('Determining optimal K using Sum of Squared Distances...')
    # optimal_k(X, k_max=10)

    K = 6
    model = KMeans(n_clusters=K, init='k-means++', max_iter=100, n_init=1)
    model.fit(X)

    print("Top terms per cluster:")
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer_tfidf.get_feature_names()
    for i in range(K):
        print("Cluster %d:" % i),
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind]),
        print

    # Prediction
    print("\n")
    print("Prediction")

    Y = vectorizer_tfidf.transform(["this movie is a horror thriller"])
    prediction = model.predict(Y)
    print(prediction)

    Y = vectorizer_tfidf.transform(["drama movies are often too long."])
    prediction = model.predict(Y)
    print(prediction)
    
