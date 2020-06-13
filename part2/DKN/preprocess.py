from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
import numpy as np
import json
import os
import random

MAX_TITLE_LENGTH = 30
MAX_ENTITY_LENGTH = 30
EMBEDDING_DIM = 300
MAX_BROWSED = 30
NEG_SAMPLE = 1


def preprocess_news_data(filename, filename_2):
    # only use news title
    print('Preprocessing news...')
    titles = []
    news_index = {}
    entity_index = {}
    all_entity = []
    with open(filename, 'r') as f:
        for l in f:
            id, category, subcategory, title, abstract, url, entity = l.strip('\n').split('\t')
            if id not in news_index:
                news_index[id] = len(news_index)
                title = title.lower()
                titles.append(title)
                entity = json.loads(entity)
                entity_list = []
                for i in entity:
                    if float(i['Confidence']) > 0.8:
                        if i['WikidataId'] not in entity_index:
                            entity_index[i['WikidataId']] = len(entity_index)
                        entity_list.append(i['WikidataId'])
                all_entity.append(entity_list)
    news_index_test = {}
    titles_test = []
    entity_test = []
    with open(filename_2, 'r') as f:
        for l in f:
            id, category, subcategory, title, abstract, url, entity = l.strip('\n').split('\t')
            entity = json.loads(entity)
            if id not in news_index:
                news_index[id] = len(news_index)
                title = title.lower()
                titles.append(title)
                entity_list = []
                for i in entity:
                    if float(i['Confidence']) > 0.8:
                        if i['WikidataId'] not in entity_index:
                            entity_index[i['WikidataId']] = len(entity_index)
                        entity_list.append(i['WikidataId'])
                all_entity.append(entity_list)
            if id not in news_index_test:
                news_index_test[id] = len(news_index_test)
                title = title.lower()
                titles_test.append(title)
                entity_list = []
                for i in entity:
                    if float(i['Confidence']) > 0.8:
                        if i['WikidataId'] not in entity_index:
                            entity_index[i['WikidataId']] = len(entity_index)
                        entity_list.append(i['WikidataId'])
                entity_test.append(entity_list)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(titles)
    word_index = tokenizer.word_index # a dict: word_index[word]=index
    print('Found %s unique news.' % len(news_index))
    print('Found %s unique tokens.' % len(word_index))
    print('Found %s unique entities.' % len(entity_index))

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
    news_entity = np.zeros((len(all_entity), MAX_ENTITY_LENGTH), dtype='int32')
    news_entity_test = np.zeros((len(entity_test), MAX_ENTITY_LENGTH), dtype='int32')
    for i, entities in enumerate(all_entity):
        k = 0
        for entity in entities:
            if k < MAX_ENTITY_LENGTH:
                news_entity[i, k] = entity_index[entity]
            k = k + 1
    for i, entities in enumerate(entity_test):
        k = 0
        for entity in entities:
            if k < MAX_ENTITY_LENGTH:
                news_entity_test[i, k] = entity_index[entity]
            k = k + 1
    news_test = np.concatenate((news_title_test, news_entity_test), axis=-1)

    return news_index, word_index, news_title, news_index_test, news_entity, entity_index, news_test


def preprocess_user_data(filename):
    print("Preprocessing user data...")
    browsed_news = []
    impression_news = []
    with open(filename, "r") as f:
        data = f.readlines()
    random.seed(212)
    random.shuffle(data)
    use_num = int(len(data) * 0.1)
    use_data = data[:use_num]
    all_browsed_news = []
    all_candidate = []
    all_label = []
    for l in use_data:
        userID, time, history, impressions = l.strip('\n').split('\t')
        history = history.split()
        browsed_news.append(history)
        impressions = [x.split('-') for x in impressions.split()]
        pos = []
        neg = []
        for news in impressions:
            if int(news[1]) == 1:
                pos.append(news[0])
            else:
                neg.append(news[0])
        for news in pos:
            all_browsed_news.append(history)
            all_candidate.append([news])
            all_label.append([1])
            neg_news = random.sample(neg, 1)
            all_browsed_news.append(history)
            all_candidate.append([neg_news[0]])
            all_label.append([0])

    random.seed(212)
    random.shuffle(all_browsed_news)
    random.seed(212)
    random.shuffle(all_candidate)
    random.seed(212)
    random.shuffle(all_label)            
    print('original behavior: ', len(browsed_news))
    print('processed behavior: ', len(all_browsed_news))
    return all_browsed_news, all_candidate, all_label


def preprocess_test_user_data(filename):
    print("Preprocessing test user data...")
    with open(filename, 'r') as f:
        data = f.readlines()
    random.seed(212)
    random.shuffle(data)
    use_num = int(len(data) * 0.1)
    use_data = data[:use_num]
    impression_index = []
    all_browsed_test = []
    all_candidate_test = []
    all_label_test = []
    for l in use_data:
        userID, time, history, impressions = l.strip('\n').split('\t')
        history = history.split()
        impressions = [x.split('-') for x in impressions.split()]
        begin = len(all_candidate_test)
        end = len(impressions) + begin
        impression_index.append([begin, end])
        for news in impressions:
            all_browsed_test.append(history)
            all_candidate_test.append([news[0]])
            all_label_test.append([int(news[1])])
    print('test samples: ', len(all_label_test))
    return impression_index,all_browsed_test, all_candidate_test, all_label_test