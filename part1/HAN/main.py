from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

from attention_rnn import TextAttBiRNN

MAX_SEQUENCE_LENGTH = 100
max_features = 70000
maxlen = 100
batch_size = 32
embedding_dims = 100
epochs = 10
category_map = {'lifestyle': 1, 'health': 2, 'news': 3, 'sports': 4, 'weather': 5, 'entertainment': 6, 'autos': 7, 'travel': 8, 'foodanddrink': 9, 'tv': 10, 'finance': 11, 'movies': 12, 'video': 13, 'music': 14, 'kids': 15, 'middleeast': 16, 'northamerica': 17}

print('Loading data...')

train_label = []
train_title = []
train_abstract = []
train_text = []
with open('../../data/news_train.tsv', 'r') as f:
    for l in f:
        category, title, abstract = l.strip('\n').split('\t')
        train_label.append(category_map[category])
        train_title.append(title)
        train_abstract.append(abstract)
        train_text.append(title + "." + abstract)
test_label = []
test_title = []
test_abstract = []
test_text = []
with open('../../data/news_test.tsv', 'r') as f:
    for l in f:
        category, title, abstract = l.strip('\n').split('\t')
        test_label.append(category_map[category])
        test_title.append(title)
        test_abstract.append(abstract)
        test_text.append(title+ "." + abstract)
all_texts = train_text + test_text
all_labels = train_label + test_label
len_train = len(train_label)
len_test = len(test_label)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_texts)
sequences = tokenizer.texts_to_sequences(all_texts)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = to_categorical(np.asarray(all_labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

x_train = data[:len_train]
y_train = labels[:len_train]
x_test = data[len_train:]
y_test = labels[len_train:]

print('Pad sequences (samples x time)...')
# x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
# x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
# print('x_train shape:', x_train.shape)
# print('x_test shape:', x_test.shape)
# labels = to_categorical(np.asarray(all_labels))
# y_train = labels[:len_train]
# y_test = labels[len_train:]

print('Build model...')
model = TextAttBiRNN(maxlen, max_features, embedding_dims, class_num=labels.shape[1], last_activation='softmax')
model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

print('Train...')
early_stopping = EarlyStopping(monitor='val_acc', patience=3, mode='max')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=[early_stopping],
          validation_data=(x_test, y_test))

print('Test...')
# result = model.predict(x_test)