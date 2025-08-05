import json
import random
from nlp_utils import tokenize, stem, bag_of_words
import numpy as np
from sklearn.naive_bayes import MultinomialNB
import pickle

with open('data/intents.json') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?', '!', '.', ',']
all_words = sorted(set(stem(w) for w in all_words if w not in ignore_words))
tags = sorted(set(tags))

X_train = [bag_of_words(x[0], all_words) for x in xy]
y_train = [tags.index(x[1]) for x in xy]

model = MultinomialNB()
model.fit(X_train, y_train)

with open('model.pkl', 'wb') as f:
    pickle.dump((model, all_words, tags), f)

print("Training completed.")
