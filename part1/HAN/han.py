# import numpy as np
# # import cv2
# import tensorflow.keras.backend as K
# import tensorflow as tf
 
# t1 = K.variable(np.array([[1, 2, 3,4],[4, 5, 6,7],[1,1,1,2]]))
# t2 = K.variable(np.array([[7,8,9,10],[10,11,12,13],[1,1,1,3]]))
# # d1 = K.concatenate([t1 , t2] , axis=1)
# # d2 = K.concatenate([t1 , t2] , axis=-1)
 
# # init = tf.global_variables_initializer()
# # with tf.Session() as sess:
# #     sess.run(init)
# #     print(sess.run(d1))
# #     print(sess.run(d2))
# z = K.variable([t1, t2])
# print(z)
# z = tf.transpose(z, perm = [1, 0, 2])
# print(z)
# # sess=tf.Session()
# # sess.run(tf.global_variables_initializer())
# # print(sess.run(z))
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
import numpy as np
import nltk
from nltk import tokenize
from attention import Attention

MAX_SEQUENCE_LENGTH = 100
MAX_SENTS = 40
max_features = 70000
maxlen = 100
batch_size = 32
EMBEDDING_DIM = 300
epochs = 10
category_map = {'lifestyle': 1, 'health': 2, 'news': 3, 'sports': 4, 'weather': 5, 'entertainment': 6, 'autos': 7, 'travel': 8, 'foodanddrink': 9, 'tv': 10, 'finance': 11, 'movies': 12, 'video': 13, 'music': 14, 'kids': 15, 'middleeast': 16, 'northamerica': 17}


def load_data():
    print("Loading data...")
    labels = []
    titles = []
    abstracts = []
    texts = []
    with open('../../data/MINDsmall_train/news.tsv', 'r') as f:
        for l in f:
            _, category, _, title, abstract, _, _ = l.strip('\n').split('\t')
            labels.append(category_map[category])
            title = title.lower()
            abstract = abstract.lower()
            titles.append(title)
            abstracts.append(abstract)
            texts.append(title + "." + abstract)
    return labels, titles, abstracts, texts

def preprocess(labels, titles, abstracts, texts):
    news = []
    labels = to_categorical(np.asarray(labels))
    for i in texts:
        sentences = tokenize.sent_tokenize(i)
        news.append(sentences)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    data = np.zeros((len(texts), MAX_SENTS, MAX_SEQUENCE_LENGTH), dtype='int32')
    for i, sentences in enumerate(news):
        for j, sent in enumerate(sentences):
            if j< MAX_SENTS:
                wordTokens = text_to_word_sequence(sent)
                k=0
                for _, word in enumerate(wordTokens):
                    if k < MAX_SEQUENCE_LENGTH and tokenizer.word_index[word] < max_features:
                        data[i,j,k] = tokenizer.word_index[word]
                        k=k+1
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)
    return word_index, data, labels

labels, titles, abstracts, texts = load_data()
word_index, data, labels = preprocess(labels, titles, abstracts, texts)
news_num = len(texts)
train_num = int(0.8 * news_num)
x_train = data[:train_num]
y_train = labels[:train_num]
x_val = data[train_num:]
y_val = labels[train_num:]
print('Shape of x_train:', x_train.shape)
print('Shape of y_train:', y_train.shape)


VECTOR_DIR = '../..//GoogleNews-vectors-negative300.bin'
print ('load word2vec as embedding...')
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

from tensorflow.keras.layers import Embedding
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)
from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from tensorflow.keras.models import Model

sentence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sentence_input)
l_lstm = Bidirectional(GRU(100, return_sequences=True))(embedded_sequences)
l_att = Attention(100)(l_lstm)
sentEncoder = Model(sentence_input, l_att)

review_input = Input(shape=(MAX_SENTS, MAX_SEQUENCE_LENGTH), dtype='int32')
review_encoder = TimeDistributed(sentEncoder)(review_input)
l_lstm_sent = Bidirectional(GRU(100, return_sequences=True))(review_encoder)
l_att_sent = Attention(100)(l_lstm_sent)
preds = Dense(labels.shape[1], activation='softmax')(l_att_sent)
model = Model(review_input, preds)

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

print("model fitting - Hierachical attention network")
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=10, batch_size=50)