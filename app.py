import nltk
import numpy as np
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Sample corpus
corpus = """Hello! How can I help you?
What is your name?
I’m a chatbot built using Python.
Tell me a joke.
Python is a great programming language.
Goodbye!"""

sent_tokens = nltk.sent_tokenize(corpus)
lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token.lower()) for token in tokens if token not in string.punctuation]

def respond(user_input):
    sent_tokens.append(user_input)
    tfidf = TfidfVectorizer(tokenizer=LemTokens, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    idx = vals.argmax()
    response = sent_tokens[idx]
    sent_tokens.pop()
    return response

# Chat loop
print("ChatBot: Ask me anything! Type 'bye' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'bye':
        print("ChatBot: Goodbye!")
        break
    else:
        print("ChatBot:", respond(user_input))
