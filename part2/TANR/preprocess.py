from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.utils import to_categorical
import numpy as np
import json
import os
import random


MAX_TITLE_LENGTH = 30
EMBEDDING_DIM = 300
MAX_BROWSED = 50
NEG_SAMPLE = 1


def preprocess_news_data(filename, filename_2):
    # only use news title
    print('Preprocessing news...')
    titles = []
    news_index = {}
    category_map = {}
    categories = []
    with open(filename, 'r') as f:
        for l in f:
            id, category, subcategory, title, abstract, url, entity = l.strip('\n').split('\t')
            if id not in news_index:
                news_index[id] = len(news_index)
                title = title.lower()
                titles.append(title)
                if category not in category_map:
                    category_map[category] = len(category_map)
                categories.append(category)
    news_index_test = {}
    titles_test = []
    with open(filename_2, 'r') as f:
        for l in f:
            id, category, subcategory, title, abstract, url, entity = l.strip('\n').split('\t')
            if id not in news_index:
                news_index[id] = len(news_index)
                title = title.lower()
                titles.append(title)
                if category not in category_map:
                    category_map[category] = len(category_map)
                categories.append(category)
            if id not in news_index_test:
                news_index_test[id] = len(news_index_test)
                title = title.lower()
                titles_test.append(title)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(titles)
    word_index = tokenizer.word_index # a dict: word_index[word]=index
    print('Found %s unique news.' % len(news_index))
    print('Found %s unique tokens.' % len(word_index))

    news_title = np.zeros((len(titles), MAX_TITLE_LENGTH), dtype='int32')
    news_title_test = np.zeros((len(titles_test), MAX_TITLE_LENGTH), dtype='int32')
    for i, title in enumerate(titles):
        wordTokens = text_to_word_sequence(title)
        k = 0
        for _, word in enumerate(wordTokens):
            if k < MAX_TITLE_LENGTH:
                news_title[i, k] = word_index[word]
                k = k + 1
    for i, title in enumerate(titles_test):
        wordTokens = text_to_word_sequence(title)
        k = 0
        for _, word in enumerate(wordTokens):
            if k < MAX_TITLE_LENGTH:
                news_title_test[i, k] = word_index[word]
                k = k + 1

    news_category = []
    news_category_test = []
    k = 0
    for category in categories:
        news_category.append(category_map[category])
        k += 1
    news_category = to_categorical(np.asarray(news_category))
    print(news_category.shape)
    print('Found unique categories: ', len(category_map))
    return news_index, word_index, news_title, news_index_test, news_title_test, news_category


def preprocess_test_user_data(filename):
    print("Preprocessing test user data...")
    with open(filename, 'r') as f:
        data = f.readlines()
    random.seed(212)
    random.shuffle(data)
    use_num = int(len(data) * 0.1)
    use_data = data[:use_num]
    impression_index = []
    user_browsed_test = []
    all_candidate_test = []
    all_label_test = []
    user_index = {}
    all_user_test = []
    for l in use_data:
        userID, time, history, impressions = l.strip('\n').split('\t')
        if userID not in user_index:
            user_index[userID] = len(user_index)
            history = history.split()
            user_browsed_test.append(history)
        impressions = [x.split('-') for x in impressions.split()]
        begin = len(all_candidate_test)
        end = len(impressions) + begin
        impression_index.append([begin, end])
        for news in impressions:
            all_user_test.append(userID)
            all_candidate_test.append([news[0]])
            all_label_test.append([int(news[1])])
    print('test samples: ', len(all_label_test))
    print('Found %s unique users.' % len(user_index))
    return impression_index, user_index, user_browsed_test, all_user_test, all_candidate_test, all_label_test


def preprocess_user_data(filename):
    print("Preprocessing user data...")
    browsed_news = []
    impression_news = []
    with open(filename, "r") as f:
        data = f.readlines()
    random.seed(212)
    random.shuffle(data)
    use_num = int(len(data) * 1)
    use_data = data[:use_num]
    for l in use_data:
        userID, time, history, impressions = l.strip('\n').split('\t')
        history = history.split()
        browsed_news.append(history)
        impressions = [x.split('-') for x in impressions.split()]
        impression_news.append(impressions)
    impression_pos = []
    impression_neg = []
    for impressions in impression_news:
        pos = []
        neg = []
        for news in impressions:
            if int(news[1]) == 1:
                pos.append(news[0])
            else:
                neg.append(news[0])
        impression_pos.append(pos)
        impression_neg.append(neg)
    all_browsed_news = []
    all_click = []
    all_unclick = []
    all_candidate = []
    all_label = []
    for k in range(len(browsed_news)):
        browsed = browsed_news[k]
        pos = impression_pos[k]
        neg = impression_neg[k]
        for pos_news in pos:
            all_browsed_news.append(browsed)
            all_click.append([pos_news])
            neg_news = random.sample(neg, NEG_SAMPLE)
            all_unclick.append(neg_news)
            all_candidate.append([pos_news]+neg_news)
            all_label.append([1] + [0] * NEG_SAMPLE)
            
    print('original behavior: ', len(browsed_news))
    print('processed behavior: ', len(all_browsed_news))
    return all_browsed_news, all_click, all_unclick, all_candidate, all_label