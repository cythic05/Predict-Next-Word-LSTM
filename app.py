import streamlit as st
import numpy as np 
import pickle as pkl
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model('next_word_lstm.h5')

with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pkl.load(handle)

def predict_next_word (model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len (token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):] # Ensure the sequence length matches max_sequ
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model. predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer. word_index.items():  
        if index == predicted_word_index:
            return word
    return None

st.title("Predict next word with the use of LSTM and EarlyStopping")
input_text = st.text_input("Enter the sequence of words")
if st.button("Predict Next Word"):
    max_seq = model.input_shape[1] + 1
    next_word = predict_next_word(model, tokenizer, input_text , max_seq)
    st.write(f'Next word: {next_word}')
