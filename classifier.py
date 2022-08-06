import re
import nltk
import numpy as np
import pandas as pd
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tensorflow.python.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.preprocessing.text import one_hot
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.layers import Embedding, LSTM, Dense, Dropout

nltk.download('stopwords')
# read train input

df = pd.read('data/train.csv')
df.dropna(inplace=True)
df.reset_index(inplace=True)
df.head(10)

x = df['title']
y = df['label']

ps = PorterStemmer()
corpus = []

# clean the data

for i in range(len(x)):
    text = x[i]
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [ps.stem(t) for t in text if t not in stopwords.words('english')]


# turn words into vectors
vocab_size = 5000
sent_len = 20
one_hot_encoded = [one_hot(x, vocab_size) for x in corpus]
# pad sequences with 0 to make them even length
one_hot_encoded = pad_sequences(one_hot_encoded, maxlen=sent_len)

# process data
x = np.array(one_hot_encoded)
y = np.array(y)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=42)

# create the model

output_features_len = 40

model = Sequential()

model.add(Embedding(vocab_size, output_features_len, input_length=sent_len))
model.add(Dropout(0.5))
model.add(LSTM(100))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# train
model.fit(x_train, y_train, validation_data=(
    x_test, y_test), batch_size=64, epochs=40)

# check predictions
predictions = model.predict(x_test)
confusion_matrix(y_test, predictions)
accuracy_score(y_test, predictions)
