import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 200
category_map = {'lifestyle': 1, 'health': 2, 'news': 3, 'sports': 4, 'weather': 5, 'entertainment': 6, 'autos': 7, 'travel': 8, 'foodanddrink': 9, 'tv': 10, 'finance': 11, 'movies': 12, 'video': 13, 'music': 14, 'kids': 15, 'middleeast': 16, 'northamerica': 17}


print("(1) loading data...")
train_label = []
train_title = []
train_abstract = []
train_text = []
with open('../data/news_train.tsv', 'r') as f:
    for l in f:
        category, title, abstract = l.strip('\n').split('\t')
        train_label.append(category_map[category])
        train_title.append(title)
        train_abstract.append(abstract)
        train_text.append(title+"."+abstract)
test_label = []
test_title = []
test_abstract = []
test_text = []
with open('../data/news_test.tsv', 'r') as f:
    for l in f:
        category, title, abstract = l.strip('\n').split('\t')
        test_label.append(category_map[category])
        test_title.append(title)
        test_abstract.append(abstract)
        test_text.append(title+"."+abstract)
all_texts = train_text + test_text
all_labels = train_label + test_label
len_train = len(train_label)
len_test = len(test_label)

print("(2) doc to var...")
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_texts)
sequences = tokenizer.texts_to_sequences(all_texts)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = to_categorical(np.asarray(all_labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

print("(3) spliting test and train")
x_train = data[:len_train]
y_train = labels[:len_train]
x_test = data[len_train:]
y_test = labels[len_train:]

print ('(4) training model...')
from tensorflow.keras.layers import Dense, Input, Flatten, Dropout
from tensorflow.keras.layers import LSTM, Embedding
from tensorflow.keras.models import Sequential

model = Sequential()
model.add(Embedding(len(word_index) + 1, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
model.add(LSTM(200, dropout=0.2, recurrent_dropout=0.2))
model.add(Dropout(0.2))
model.add(Dense(labels.shape[1], activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])
print(model.metrics_names)
model.fit(x_train, y_train, epochs=2, batch_size=128)
print ('(5) testing model...')
print (model.evaluate(x_test, y_test))