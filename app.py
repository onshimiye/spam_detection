import streamlit as st
import sys
import numpy as np
import pandas as pd
from numpy import zeros
import tensorflow as tf
# helps in text preprocessing
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# helps in model building
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import Adam


st.title('Welcome to the Spam Detection Aplication')

st.header('Please provide your email body below')

email = st.text_input('Email you will like to predict if it is spam or not')

t = Tokenizer()

encoded_test = t.texts_to_sequences(email)

# pad documents to a max length of 6 words
max_length = 6  
padded_test = pad_sequences(encoded_test, maxlen=max_length, padding='post')
print(padded_test)