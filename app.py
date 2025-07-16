from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
import pickle
import json
import random
from datetime import datetime
import os
import re

app = Flask(__name__)

# Configuration
MAX_LEN = 20
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

# Global variables
MODELS_LOADED = False
data = None
model = None
sentiment_model = None
tokenizer = None
lbl_encoder = None

def initialize_app():
    global MODELS_LOADED, data, model, sentiment_model, tokenizer, lbl_encoder
    
    try:
        # Load dataset first
        with open('uem_kolkata_chatbot_dataset.json', 'r', encoding='utf-8') as file:
            data = json.load(file)
            if not data.get('intents'):
                raise ValueError("Dataset is empty or malformed")
        
        # Load models
        model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'chatbot_model.keras'))
        sentiment_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'sentiment_model.keras'))
        
        with open(os.path.join(MODEL_DIR, 'tokenizer.pickle'), 'rb') as handle:
            tokenizer = pickle.load(handle)
        
        with open(os.path.join(MODEL_DIR, 'label_encoder.pickle'), 'rb') as ecn_file:
            lbl_encoder = pickle.load(ecn_file)
            
        MODELS_LOADED = True
        print("All models and data loaded successfully")
        
    except Exception as e:
        print(f"Initialization error: {str(e)}")
        # Try training models if loading fails
        from train_chatbot import train_and_save_models
        train_and_save_models()
        initialize_app()  # Try again after training

# Initialize the app
initialize_app()

def sanitize_input(text):
    text = re.sub(r'[^\w\s.,!?]', '', text.strip())
    return text[:500]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_bot_response():
    try:
        if not request.is_json:
            return jsonify({
                'status': 'error',
                'response': "Please send your request in JSON format."
            }), 400
            
        user_input = request.get_json().get('message', '')
        if not user_input:
            return jsonify({
                'status': 'error',
                'response': "Please enter a message."
            }), 400

        user_input = sanitize_input(user_input)
        
        # Tokenize and pad the input
        sequence = tokenizer.texts_to_sequences([user_input])
        if not sequence or not sequence[0]:
            return jsonify({
                'status': 'success',
                'response': "I didn't understand that. Could you rephrase your question?",
                'sentiment': 'neutral',
                'intent': 'unknown'
            })
            
        padded = pad_sequences(sequence, maxlen=MAX_LEN)
        
        # Get predictions
        intent_pred = model.predict(padded, verbose=0)
        sentiment_pred = sentiment_model.predict(padded, verbose=0)
        
        intent_tag = lbl_encoder.inverse_transform([np.argmax(intent_pred)])[0]
        sentiment = ['negative', 'neutral', 'positive'][np.argmax(sentiment_pred)]
        
        # Find matching intent
        response = "I'm not sure how to respond to that. Could you please rephrase your question about UEM Kolkata?"
        for intent in data['intents']:
            if intent['tag'] == intent_tag:
                response = random.choice(intent['responses'])
                
                # Add sentiment-specific prefixes if not feedback intent
                if intent_tag not in ['negative_feedback', 'positive_feedback']:
                    if sentiment == 'negative':
                        prefixes = ["I'm sorry you feel that way. ", "I apologize. "]
                        response = random.choice(prefixes) + response
                    elif sentiment == 'positive':
                        prefixes = ["Great! ", "Happy to help! "]
                        response = random.choice(prefixes) + response
                break

        return jsonify({
            'status': 'success',
            'response': response,
            'sentiment': sentiment,
            'intent': intent_tag
        })
        
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({
            'status': 'error',
            'response': "Sorry, I'm having trouble processing your request. Please try again later."
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)