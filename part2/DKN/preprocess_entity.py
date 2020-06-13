import pandas as pd
import json


def preprocess_embedding():
    embedding = {}
    data = pd.read_csv('../../data/MINDsmall_train/entity_embedding.vec')
    for line in data.values: 
        line = line[0].split('\t')[:-1]
        id = line[0]
        vec = list(map(float,line[1:]))
        if id not in embedding:
            embedding[id] = vec
    data = pd.read_csv('../../data/MINDsmall_dev/entity_embedding.vec')
    for line in data.values: 
        line = line[0].split('\t')[:-1]
        id = line[0]
        vec = list(map(float,line[1:]))
        if id not in embedding:
            embedding[id] = vec
    json_str = json.dumps(embedding)
    with open('embedding.json', 'w') as json_file:
        json_file.write(json_str)


preprocess_embedding()