# !pip install transformers

import torch
from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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
X = [preprocess(clean_text(x)) for x in X]

X_ = []
Y_ = []

for x,y in zip(X,Y):
    if len(x.split()) > 0:
        X_.append(x)
        Y_.append(y)

X_train, X_test, Y_train, Y_test = train_test_split(X_, Y_, test_size = 0.1, random_state = 19720)


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).
    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # safe to call this function even if cuda is not available
    if is_tf_available():
        import tensorflow as tf
        tf.random.set_seed(seed)

##################################################
##################################################
##################################################
set_seed(1)

# the model we gonna train, base uncased BERT
# check text classification models here: https://huggingface.co/models?filter=text-classification
model_name = "mental/mental-bert-base-uncased"
# max sequence length for each document/sentence sample
max_length = 256
# load the tokenizer
tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case=True)

# call the function
# tokenize the dataset, truncate when passed `max_length`, 
# and pad with 0's when less than `max_length`
train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=max_length)
valid_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=max_length)

class DiseaseDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)

# convert our tokenized data into a torch Dataset
train_dataset = DiseaseDataset(train_encodings, Y_train)
valid_dataset = DiseaseDataset(valid_encodings, Y_test)
# load the model and pass to CUDA
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=4).to("cuda")

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # calculate accuracy using sklearn's function
    acc = accuracy_score(labels, preds)
    return {
      'accuracy': acc,
    }

training_args = TrainingArguments(
    output_dir='../results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=8,  # batch size per device during training
    per_device_eval_batch_size=20,   # batch size for evaluation
    warmup_steps=250,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    load_best_model_at_end=True,     # load the best model when finished training (default metric is loss)
    # but you can specify `metric_for_best_model` argument to change to accuracy or other metric
    logging_steps=200,               # log & save weights each logging_steps
    save_steps=400,
    evaluation_strategy="steps",     # evaluate each `logging_steps`
)

trainer = Trainer(
    model=model,                         # the instantiated Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=valid_dataset,          # evaluation dataset
    compute_metrics=compute_metrics,     # the callback that computes metrics of interest
)
# train the model
trainer.train()
# evaluate the current model after training
trainer.evaluate()
# saving the fine tuned model & tokenizer
model_path = "../MentalDiseases-bert-base-uncased"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)