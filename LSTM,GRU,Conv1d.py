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


from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, GRU
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import re

import numpy

numpy.random.seed(7)


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

X = [preprocess(clean_text(x)) for x in X]


X_ = []
Y_ = []

for x,y in zip(X,Y):
    if len(x.split()) > 0:
        X_.append(x)
        Y_.append(y)

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

lengths = [len(x.split()) for x in X_]
  
mu, std = norm.fit(lengths) 

plt.hist(lengths, bins=25, edgecolor='black', range=(0, 250))
plt.title('Histogram of sequence lengths')
plt.xlabel('Length of sequence')
plt.ylabel('Frequency')

max_fatures = 5000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')

tokenizer.fit_on_texts(X_)
X_t = tokenizer.texts_to_sequences(X_)
X_t = pad_sequences(X_t, 80)

import tensorflow
X_train, X_test, Y_train, Y_test = train_test_split(X_t, Y_, test_size = 0.1, random_state = 19720)

Y_test_ = tensorflow.keras.utils.to_categorical(Y_test, num_classes = 4)
Y_train_ = tensorflow.keras.utils.to_categorical(Y_train, num_classes = 4)


embed_dim = 32
lstm_out = 16

model = Sequential()

model.add(Embedding(max_fatures, embed_dim,input_length = X_t.shape[1]))
model.add(SpatialDropout1D(0.4))

model.add(LSTM(lstm_out, dropout=0.2))

model.add(Dense(4,activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())


batch_size = 32
model.fit(X_train, Y_train_, epochs = 1, batch_size=batch_size, verbose = 1, validation_data= (X_test, Y_test_))


from sklearn import svm
from sklearn.metrics import plot_confusion_matrix
# from mlxtend.plotting import plot_decision_regions

y_pred = model.predict(X_test)

y_pred = np.argmax(y_pred, axis = 1)

%matplotlib notebook
import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, y_pred)
f = sns.heatmap(cm, annot=True, fmt='d')

from sklearn.metrics import classification_report

print(classification_report(Y_test, y_pred, target_names = target_names.values()))



from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.callbacks import EarlyStopping


input1 = Input(shape=(80,))
x = Embedding(input_dim=2000, output_dim=32, input_length=80)(input1)
x = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(x)
x = MaxPooling1D(pool_size=2)(x)
x = Dropout(0.5)(x)
x = Flatten()(x)
x = Dense(30, activation='sigmoid')(x)
x = Dropout(0.5)(x)
x = Dense(5, activation='sigmoid')(x)
x = Dropout(0.5)(x)
output1 = Dense(4, activation='sigmoid')(x)

model = Model(input1, output1)

model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics=['acc'])
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

print(model.summary())

history = model.fit(X_train, Y_train_, epochs=2, 
                    validation_data=(X_test, Y_test_), 
                    callbacks=[es])



from sklearn import svm
from sklearn.metrics import plot_confusion_matrix
# from mlxtend.plotting import plot_decision_regions

y_pred = model.predict(X_test)

y_pred = np.argmax(y_pred, axis = 1)

%matplotlib notebook
import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, y_pred)
f = sns.heatmap(cm, annot=True, fmt='d')



from sklearn import svm
from sklearn.metrics import plot_confusion_matrix
# from mlxtend.plotting import plot_decision_regions

y_pred = model.predict(X_test)

y_pred = np.argmax(y_pred, axis = 1)

%matplotlib notebook
import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, y_pred)
f = sns.heatmap(cm, annot=True, fmt='d')                    


embed_dim = 128
lstm_out = 196

model1 = Sequential()

model1.add(Embedding(max_fatures, embed_dim,input_length = X_t.shape[1]))
model1.add(SpatialDropout1D(0.4))

model1.add(GRU(lstm_out, dropout=0.2))

model1.add(Dense(4,activation='softmax'))

model1.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model1.summary())


batch_size = 32
model1.fit(X_train, Y_train_, epochs = 7, batch_size=batch_size, verbose = 1, validation_data= (X_test, Y_test_))