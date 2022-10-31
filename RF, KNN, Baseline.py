import pandas as pd

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA

from tqdm.notebook import tqdm
from collections import Counter
import re
import string
import random

from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score

import nltk
from nltk.corpus import stopwords
from  nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


nltk.download('stopwords')


# TEXT CLENAING
TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"


stop_words = stopwords.words("english")
stemmer = SnowballStemmer("english")

def preprocess(text, stem=False):
    # Remove link,user and special characters
    text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)
    return " ".join(tokens)

def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    if isinstance(text, str) == False:
        return 'None'
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

def text_processing(text):
    
    #Remove twitter handlers
    text = re.sub('@[^\s]+','', text)

    #remove hashtags
    text = re.sub(r'\B#\S+','', text)
    
    # Remove URLS
    text = re.sub(r"http\S+", "", text)

    # Remove all the special characters
    text = ' '.join(re.findall(r'\w+', text))

    # Remove all single characters
    text = re.sub(r'\s+[a-zA-Z]\s+', '', text)

    # Substituting multiple spaces with single space
    text = re.sub(r'\s+', ' ', text, flags=re.I) 
    
    return text


adhd = pd.read_csv('adhd.csv')
adhd.dropna(inplace = True)

depression = pd.read_csv('Depression.csv')
depression.dropna(inplace = True)
512
ocd = pd.read_csv('OCD.csv')
ocd.dropna(inplace = True)

ptsd = pd.read_csv('PTSD.csv')
ptsd.dropna(inplace = True)


adhd_body  = adhd['body']
depression_body = depression['body']
ocd_body = ocd['body']
ptsd_body = ptsd['body']

dataset = pd.DataFrame()

for idx, body in enumerate([adhd_body, depression_body, ocd_body, ptsd_body]):
    data = pd.DataFrame()
    data['body'] = body
    data['label'] = idx
    dataset = dataset.append(data)

X = list(dataset['body'])
Y = list(dataset['label'])

X_ = []
Y_ = []

for x,y in zip(X,Y):
    if len(x.split()) > 0:
        X_.append(x)
        Y_.append(y)    

X = [preprocess(clean_text(x)) for x in X]        


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state = 19720)


from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(X, ngram_range = (3,3), max_features = 5000)
X_train_ = vectorizer.fit_transform(X_train).toarray()
X_test_ = vectorizer.transform(X_test).toarray()

#Dependencies
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# Neural network
model = Sequential()
model.add(Dense(64, input_dim=5000, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


Y_test_ = tf.keras.utils.to_categorical(Y_test, num_classes = 4)
Y_train_ = tf.keras.utils.to_categorical(Y_train, num_classes = 4)


history = model.fit(X_train_, Y_train_, epochs=3, batch_size=64, validation_data = (X_test_, Y_test_))


from sklearn import svm
from sklearn.metrics import plot_confusion_matrix
# from mlxtend.plotting import plot_decision_regions

y_pred = model.predict(X_test_)

y_pred = np.argmax(y_pred, axis = 1)

%matplotlib notebook
import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, y_pred)
f = sns.heatmap(cm, annot=True, fmt='d')


from sklearn.metrics import classification_report

target_names  = {
    0 : 'adhd',
    1 : 'depression',
    2 : 'ocd',
    3 : 'ptsd'
    }

print(classification_report(Y_test, y_pred, target_names = target_names.values()))


# WordCloud


import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.callbacks import EarlyStopping
%matplotlib inline

all_text = X_
all_text = ' '.join(all_text)


wordcloud = WordCloud(width=2560, height=1440, 
                    background_color='black',
                    min_font_size=10)
word_cloud = wordcloud.generate(all_text)


plt.figure(figsize=(16,9))
plt.imshow(word_cloud)
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.show()


# KNN

from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=3)

neigh.fit(X_train_, Y_train)

y_pred = neigh.predict(X_test_)
accuracy_score(y_pred, Y_test)


test_data = vectorizer.transform(['stiffness']).toarray()




from sklearn.metrics import classification_report

print(classification_report(Y_test, y_pred))