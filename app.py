import streamlit as st
import pickle
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import pad_sequences
#load lstm model

model = load_model('collab/LSTM47.keras')

#load tokenizer
with open("collab/tokenizer.pickle", 'rb') as handle:
    tokenizer = pickle.load(handle)

#predict function

def predict_next_word(model, tokenizer, text, max_sequence_length):
    text_token = tokenizer.texts_to_sequences([text])[0]
    if len(text_token)>= max_sequence_length:
        text_token = text_token[-(max_sequence_length):]
    text_token = pad_sequences([text_token], maxlen=max_sequence_length, padding='pre')

    predicted = model.predict(text_token)
    predict_word_inedex = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predict_word_inedex:
            return word
    return None 

#streamlit app

st.title("NEXT WORD PREDICTION WITH LSTM")
input_text = st.text_input("Enter the sequence of word", "To be or not to be")
if st.button("Predict next word"):
    max_sequence_length = model.input_shape[1] + 1
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_length)
    st.write(f"NEXT WORD: {next_word}")
