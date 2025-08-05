import pickle
import json
import random
from nlp_utils import tokenize, bag_of_words

with open('model.pkl', 'rb') as f:
    model, all_words, tags = pickle.load(f)

with open('data/intents.json') as f:
    intents = json.load(f)

def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    tag_index = model.predict([X])[0]
    tag = tags[tag_index]

    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

# Chat loop
if __name__ == "__main__":
    print("Bot: Hello! Type 'quit' to exit.")
    while True:
        msg = input("You: ")
        if msg.lower() == "quit":
            break
        response = get_response(msg)
        print("Bot:", response)

