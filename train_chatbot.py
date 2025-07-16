import json
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM #type: ignore
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from sklearn.preprocessing import LabelEncoder
import pickle
import os

def train_and_save_models():
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Load and prepare data
    try:
        with open(r'D:\innovation\University Assistant Project\uem_kolkata_chatbot_dataset.json', 'r', encoding='utf-8') as file:
            data = json.load(file)
        print("✅ JSON loaded successfully!")
    except Exception as e:
        print("❌ Failed to load JSON:", e)
        exit()

    vocab_size = 3000
    training_sentences = []
    training_labels = []
    sentiments = []
    
    for intent in data['intents']:
        for pattern in intent['patterns']:
            training_sentences.append(pattern)
            training_labels.append(intent['tag'])
            sentiments.append(intent['sentiment'])
    
    # Encode labels
    lbl_encoder = LabelEncoder()
    training_labels = lbl_encoder.fit_transform(training_labels)
    
    # Tokenize text
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(training_sentences)
    sequences = tokenizer.texts_to_sequences(training_sentences)
    padded_sequences = pad_sequences(sequences, maxlen=20, truncating='post')
    
    # Encode sentiments
    sentiment_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
    encoded_sentiments = np.array([sentiment_mapping[s] for s in sentiments])
    
    # Intent model
    intent_model = Sequential([
    Embedding(vocab_size, 64, input_length=20),
    LSTM(64, return_sequences=False),
    Dense(32, activation='relu'),
    Dense(len(lbl_encoder.classes_), activation='softmax')
    ])
    intent_model.compile(loss='sparse_categorical_crossentropy',
                       optimizer=Adam(0.001),
                       metrics=['accuracy'])
    intent_model.fit(padded_sequences, np.array(training_labels), epochs=500, verbose=1)
    
    # Sentiment model
    sentiment_model = Sequential([
        Embedding(vocab_size, 64, input_length=20),
        GlobalAveragePooling1D(),
        Dense(16, activation='relu'),
        Dense(3, activation='softmax')
    ])
    sentiment_model.compile(loss='sparse_categorical_crossentropy',
                          optimizer=Adam(0.001),
                          metrics=['accuracy'])
    sentiment_model.fit(padded_sequences, encoded_sentiments, epochs=500, verbose=1)
    
    # Save models
    intent_model.save('models/chatbot_model.keras')
    sentiment_model.save('models/sentiment_model.keras')
    
    with open('models/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open('models/label_encoder.pickle', 'wb') as ecn_file:
        pickle.dump(lbl_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    train_and_save_models()