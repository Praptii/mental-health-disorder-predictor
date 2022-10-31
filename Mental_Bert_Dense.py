from transformers import BertModel
from transformers import AutoTokenizer, AutoModel
import torch


import pandas as pd
# from Bio import SeqIO
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


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

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

disease_dict = {
    0 : 'adhd',
    1 : 'depression',
    2 : 'ocd',
    3 : 'ptsd'
}

adhd = pd.read_csv('../ADHD')
adhd.dropna(inplace = True)

depression = pd.read_csv('../Depression')
depression.dropna(inplace = True)

ocd = pd.read_csv('../OCD')
ocd.dropna(inplace = True)

ptsd = pd.read_csv('../PTSD')
ptsd.dropna(inplace = True)

print(adhd.head())

print(ptsd.head())
print(ocd.head())
print(depression.head())

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

print('Preprocessing data ')
X_ = [preprocess(clean_text(x)) for x in X[:1000]]

X_train, X_test, Y_train, Y_test = train_test_split(X_, Y[:1000], test_size = 0.1, random_state = 19720)

tokenizer = AutoTokenizer.from_pretrained("mental/mental-bert-base-uncased")

class CustomBERTModel(torch.nn.Module):
    def __init__(self):
          super(CustomBERTModel, self).__init__()
          self.bert = AutoModel.from_pretrained("mental/mental-bert-base-uncased")
          ### New layers:
          self.linear1 = torch.nn.Linear(256,32)
          self.linear2 = torch.nn.Linear(32, 4)

    def forward(self, ids, mask):
          sequence_output, pooled_output = self.bert(
               ids, 
               attention_mask=mask).to_tuple()
         
          
          linear1_output = self.linear1(sequence_output[:,0,:].contiguous().view(-1,32)) ## extract the 1st token's embeddings
          linear2_output = self.linear2(linear1_output) 

          return linear2_output

tokenizer = AutoTokenizer.from_pretrained("mental/mental-bert-base-uncased")
model = CustomBERTModel() # You can pass the parameters if required to have more flexible model
# model.cuda() ## can be gpu

criterion = torch.nn.CrossEntropyLoss() ## If required define your own criterion
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))


optimizer.zero_grad()   
encoding = tokenizer.batch_encode_plus(X_train[:120], return_tensors='pt', padding=True, truncation=True,max_length=50, add_special_tokens = True)
input_ids = encoding['input_ids']
attention_mask = encoding['attention_mask']
outputs = model(input_ids, attention_mask)
targets = to_categorical(Y_train[:120], 4)

targets = torch.from_numpy(targets).long()

loss = criterion(outputs, targets)
loss.backward()
optimizer.step()
        
