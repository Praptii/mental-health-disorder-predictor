from transformers import BertTokenizerFast, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups

model_path = "New folder"
max_length = 512


model = BertForSequenceClassification.from_pretrained(model_path, num_labels=4).to("cuda")
tokenizer = BertTokenizerFast.from_pretrained(model_path)

def get_prediction(text):
    target_names  = {
    0 : 'adhd',
    1 : 'depression',
    2 : 'ocd',
    3 : 'ptsd'
    }
    # prepare our text into tokenized sequence
    inputs = tokenizer(text, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to("cuda")
    # perform inference to our model
    outputs = model(**inputs)
    # get output probabilities by doing softmax
    probs = outputs[0].softmax(1)
    # executing argmax function to get the candidate label
    return target_names[probs.argmax().item()]

# Example #1
text = """
I have a lot of repressed memories. I have traumatic past because of war and violence.
"""
print(get_prediction(text))
# Example #2
text = """
I cannot meet deadlines and cannot start early. I cannot pay attention to things and I cannot sit quietly.
"""
print(get_prediction(text))