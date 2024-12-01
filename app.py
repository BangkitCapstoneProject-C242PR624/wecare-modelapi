from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import random
import nltk
import numpy as np
import string
import tensorflow as tf
from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer

# Inisialisasi Flask app
app = Flask(__name__)

CORS(app)

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

# Memuat model sekali saat aplikasi dijalankan
model = load_model('chatbot_model.h5')

# Fungsi prediksi kelas (tag) berdasarkan input
def predict_class(sentence):
    p = bow(sentence, all_words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.3
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [classes[r[0]] for r in results]
    return return_list

# Fungsi untuk mendapatkan respons berdasarkan tag
def get_response(intents_list, intents_json):
    if not intents_list:
        return "I'm sorry, I couldn't understand that. Could you please rephrase?"
    tag = intents_list[0]
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "I'm sorry, I don't have an appropriate response for that."

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
    app.run(host='0.0.0.0', port=5000)