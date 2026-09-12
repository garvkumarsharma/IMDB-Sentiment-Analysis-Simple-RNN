import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model


# Cache the word index so it's only loaded once, not on every Streamlit rerun
@st.cache_resource
def get_word_index():
    word_index = imdb.get_word_index()
    reverse_word_index = {value: key for key, value in word_index.items()}
    return word_index, reverse_word_index


# Cache the model so it's only loaded from disk once, not on every rerun
@st.cache_resource
def get_model():
    model = load_model('simple_rnn_imdb.h5')
    return model


word_index, reverse_word_index = get_word_index()
model = get_model()


# Helper Functions

# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])


# Must match the num_words / max_features used when training the model,
# since the Embedding layer's vocab size (input_dim) is fixed to this value.
max_features = 10000


# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = []
    for word in words:
        idx = word_index.get(word, 2) + 3
        # word_index covers the FULL ~88k-word vocabulary, but the model's
        # Embedding layer only has max_features entries. Any index that
        # would fall outside that range must be treated as OOV (2).
        if idx >= max_features:
            idx = 2
        encoded_review.append(idx)
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


# Prediction Function
def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]


# Streamlit App

st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative')

# User Input
user_input = st.text_area('Movie Review')

if st.button('Classify'):
    if user_input.strip() == '':
        st.warning('Please enter a movie review before classifying.')
    else:
        sentiment, score = predict_sentiment(user_input)

        # Display the result
        st.write(f'Sentiment : {sentiment}')
        st.write(f'Prediction Score : {score}')
else:
    st.write('Please enter a movie review')