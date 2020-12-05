import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle


st.title('Welcome to the Spam Detection Aplication')

st.header('Please provide your email body below')

email = st.text_input('Email you will like to predict if it is spam or not')


s_model = tf.keras.models.load_model("spam_detection")

with open('spam_detection/tokenizer.pkl', 'rb') as input:
    tokener = pickle.load(input)

if email:

    email = [email]
    encoded_email = tokener.texts_to_sequences(email)

    # pad documents to a max length of 6 words
    max_length = 6  
    padded_email = pad_sequences(encoded_email, maxlen=max_length, padding='post')

    pred = s_model.predict(padded_email) 

    if pred > 0.5:
        st.info('This email is a spam! With a probability of: {}'.format(pred[0][0]) )
    else:
        st.info('This email is not a spam! With a probability of: {}'.format(pred[0][0]) )

    st.write('End!')
