from flask import Flask, request, jsonify
import json
import random
import nltk
import numpy as np
import string
import tensorflow as tf
from wecare import predict_class
from wecare import get_response
from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer
from flask import Flask, request, jsonify
from flask import make_response
from flask_cors import CORS

# Inisialisasi Flask app
app = Flask(__name__)

CORS(app, resources={r"/chat": {"origins": "http://127.0.0.1:5500"}})

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Memuat data dari dataset
with open('dataset.json', 'r') as file:
    data = json.load(file)

# Inisialisasi lemmatizer dan mempersiapkan dataset
lemmatizer = WordNetLemmatizer()
all_words = []
classes = []
documents = []

# Tokenisasi dan lemmatization untuk setiap pola
for intent in data['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        all_words.extend(word_list)
        documents.append((word_list, intent['tag']))
    if intent['tag'] not in classes:
        classes.append(intent['tag'])

# Persiapkan vocabulary
all_words = [lemmatizer.lemmatize(word.lower()) for word in all_words if word not in string.punctuation]
all_words = sorted(set(all_words))
classes = sorted(set(classes))

# Fungsi untuk membersihkan dan memproses kalimat input
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Fungsi untuk membuat bag-of-words (BOW)
def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
    return np.array(bag)

# Fungsi prediksi kelas (tag) berdasarkan input
def predict_class(sentence):
    p = bow(sentence, all_words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.3
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [classes[r[0]] for r in results]
    return return_list

model = load_model('chatbot_model.h5')

# Fungsi untuk mendapatkan respons berdasarkan tag
def get_response(intents_list, intents_json):
    if not intents_list:
        return "I'm sorry, I couldn't understand that. Could you please rephrase?"
    tag = intents_list[0]
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "I'm sorry, I don't have an appropriate response for that."

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "http://127.0.0.1:5500"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Max-Age"] = "3600"
    return response


@app.before_request
def log_request_info():
    print(f"Headers: {request.headers}")
    print(f"Body: {request.get_data()}")

@app.after_request
def log_response_info(response):
    print(f"Response headers: {response.headers}")
    return response


# Endpoint API untuk chat
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    
    if not user_input:
        return jsonify({"error": "No message provided"}), 400
    
    # Menggunakan model untuk memprediksi kelas
    intents = predict_class(user_input)
    response = get_response(intents, data)
    
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
