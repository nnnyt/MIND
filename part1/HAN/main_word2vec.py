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
EMBEDDING_DIM = 300
epochs = 10
category_map = {'lifestyle': 1, 'health': 2, 'news': 3, 'sports': 4, 'weather': 5, 'entertainment': 6, 'autos': 7, 'travel': 8, 'foodanddrink': 9, 'tv': 10, 'finance': 11, 'movies': 12, 'video': 13, 'music': 14, 'kids': 15, 'middleeast': 16, 'northamerica': 17}

print('(1) Loading data...')

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

print("(2) doc to var")
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

VECTOR_DIR = '../..//GoogleNews-vectors-negative300.bin'
print ('(4) load word2vec as embedding...')

import gensim
from tensorflow.keras.utils import plot_model
w2v_model = gensim.models.KeyedVectors.load_word2vec_format(VECTOR_DIR, binary=True)
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
not_in_model = 0
in_model = 0
for word, i in word_index.items(): 
    if word in w2v_model:
        in_model += 1
        embedding_matrix[i] = np.asarray(w2v_model[word], dtype='float32')
    else:
        not_in_model += 1
print(str(in_model) + ' in w2v model')
print(str(not_in_model)+' words not in w2v model')
# from tensorflow.keras.layers import Embedding
# embedding_layer = Embedding(len(word_index) + 1,
#                             EMBEDDING_DIM,
#                             weights=[embedding_matrix],
#                             input_length=MAX_SEQUENCE_LENGTH,
#                             trainable=True)#test

print('(5) Build model...')
model = TextAttBiRNN(maxlen, max_features, EMBEDDING_DIM, embedding_matrix, class_num=labels.shape[1], last_activation='softmax')
model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

print('Train...')
early_stopping = EarlyStopping(monitor='val_acc', patience=3, mode='max')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=[early_stopping],
          validation_data=(x_test, y_test))

# print('Test...')
# result = model.predict(x_test)