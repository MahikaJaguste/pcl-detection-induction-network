import configparser
import os
import re
import string
import pickle
import copy
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from fastNLP import Vocabulary
from dataset import Dataset
from dataloader import TrainDataLoader
from utils import padding, batch_padding, padding_2


def _parse_list(data_path, list_name):
    domain = set()
    with open(os.path.join(data_path, list_name), 'r', encoding='utf-8') as f:
        for line in f:
            domain.add(line.strip('\n'))
    return domain


def get_domains(data_path, filtered_name, target_name):
    all_domains = _parse_list(data_path, filtered_name)
    test_domains = _parse_list(data_path, target_name)
    # train_domains = all_domains - test_domains
    train_domains = all_domains
    print('train domains', len(train_domains), 'test_domains', len(test_domains))
    return sorted(list(train_domains)), sorted(list(test_domains))

## single file
def _parse_data(data_path, filename):
    degree_0 = {
        'filename': filename,
        'data': [],
        'target': []
    }
    degree_1 = {
        'filename': filename,
        'data': [],
        'target': []
    }
    degree_2 = {
        'filename': filename,
        'data': [],
        'target': []
    }
    degree_3 = {
        'filename': filename,
        'data': [],
        'target': []
    }
    degree_4 = {
        'filename': filename,
        'data': [],
        'target': []
    }
    real_filename = filename + ".csv"
    df = pd.read_csv(os.path.join(data_path, real_filename)) # encoding='utf-8'
    for index, row in df.iterrows():
        if(row['score']==0):
            degree_0['data'].append(row['tweet'])
            degree_0['target'].append(0)
        if(row['score']==1):
            degree_1['data'].append(row['tweet'])
            degree_1['target'].append(1)
        if(row['score']==2):
            degree_2['data'].append(row['tweet'])
            degree_2['target'].append(2)
        if(row['score']==3):
            degree_3['data'].append(row['tweet'])
            degree_3['target'].append(3)
        if(row['score']==4):
            degree_4['data'].append(row['tweet'])
            degree_4['target'].append(4)
    # check
    print(filename, 'degree_0', len(degree_0['data']), 'degree_1', len(degree_1['data']), 'degree_2', len(degree_2['data']),\
        'degree_3', len(degree_3['data']), 'degree_4', len(degree_4['data']),)
    return degree_0, degree_1, degree_2, degree_3, degree_4


def _process_data(data_dict):
    for i in range(len(data_dict['data'])):
        text = data_dict['data'][i]
        # ignore string.punctuation
        text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
        # string.whitespace -> space
        text = re.sub('[%s]' % re.escape(string.whitespace), ' ', text)
        # lower case
        text = text.lower()
        # split by whitespace
        text = text.split()
        # replace
        data_dict['data'][i] = text
    return data_dict


def _get_data(data_path, domains, usage):
    # usage in ['train', 'dev', 'test']
    data = {}
    for domain in domains:
        filename = '.'.join([domain, usage])
        degree_0, degree_1, degree_2, degree_3, degree_4 = _parse_data(data_path, filename)
        degree_0 = _process_data(degree_0)
        degree_1 = _process_data(degree_1)
        degree_2 = _process_data(degree_2) 
        degree_3 = _process_data(degree_3)
        degree_4 = _process_data(degree_4)
        data[filename] = {'degree_0': degree_0,
                          'degree_1': degree_1,
                          'degree_2': degree_2,
                          'degree_3': degree_3,
                          'degree_4': degree_4}
    return data


def get_train_data(data_path, domains):
    train_data = _get_data(data_path, domains, 'train')
    print('train data', len(train_data))
    return train_data


def _combine_data(support_data, data):
    # support -> dev, test
    for key in data:
        key_split = key.split('.')[0:-1] + ['support']
        support_key = '.'.join(key_split)
        for value in data[key]:
            data[key][value]['support_data'] = copy.deepcopy(support_data[support_key][value]['data'])
            data[key][value]['support_target'] = copy.deepcopy(support_data[support_key][value]['target'])
    return data


def get_test_data(data_path, domains):
    # get dev, test data
    support_data = _get_data(data_path, domains, 'support')
    dev_data = _get_data(data_path, domains, 'dev')
    test_data = _get_data(data_path, domains, 'test')

    # support -> dev, test
    dev_data = _combine_data(support_data, dev_data)
    test_data = _combine_data(support_data, test_data)

    print('dev data', len(dev_data), 'test data', len(test_data))
    return dev_data, test_data


def get_vocabulary(data, min_freq):
    # train data -> vocabulary
    vocabulary = Vocabulary(min_freq=min_freq, padding='<pad>', unknown='<unk>')
    for filename in data:
        for value in data[filename]:
            for word_list in data[filename][value]['data']:
                vocabulary.add_word_lst(word_list)
    vocabulary.build_vocab()
    print('vocab size', len(vocabulary), 'pad', vocabulary.padding_idx, 'unk', vocabulary.unknown_idx)
    return vocabulary


def _idx_text(text_list, vocabulary):
    for i in range(len(text_list)):
        for j in range(len(text_list[i])):
            text_list[i][j] = vocabulary.to_index(text_list[i][j])
    return text_list


def idx_all_data(data, vocabulary):
    for filename in data:
        for value in data[filename]:
            for key in data[filename][value]:
                if key in ['data', 'support_data']:
                    data[filename][value][key] = _idx_text(data[filename][value][key], vocabulary)
    return data


def get_train_loader(train_data, support, query, pad_idx):
    batch_size = support + query
    train_loaders = {}
    for filename in train_data:
        degree_0_dl = DataLoader(Dataset(train_data[filename]['degree_0'], pad_idx), batch_size=batch_size, shuffle=True, drop_last=False, **kwargs)
        degree_1_dl = DataLoader(Dataset(train_data[filename]['degree_1'], pad_idx), batch_size=batch_size, shuffle=True, drop_last=False, **kwargs)
        degree_2_dl = DataLoader(Dataset(train_data[filename]['degree_2'], pad_idx), batch_size=batch_size, shuffle=True, drop_last=False, **kwargs)
        degree_3_dl = DataLoader(Dataset(train_data[filename]['degree_3'], pad_idx), batch_size=batch_size, shuffle=True, drop_last=False, **kwargs)
        degree_4_dl = DataLoader(Dataset(train_data[filename]['degree_4'], pad_idx), batch_size=batch_size, shuffle=True, drop_last=False, **kwargs)
        if min(len(degree_0_dl), len(degree_1_dl), len(degree_2_dl), len(degree_3_dl), len(degree_4_dl)) > 0:
            train_loaders[filename] = {'degree_0': degree_0_dl,
                                        'degree_1': degree_1_dl,
                                        'degree_2': degree_2_dl,
                                        'degree_3': degree_3_dl,
                                        'degree_4': degree_4_dl}

    # print("------------------ ", type(train_loaders['homeless.train']['degree_1']))
    # print("------------------ ", type(enumerate(train_loaders['homeless.train']['degree_1'])))
    # print("------------------ ", next(iter(train_loaders['homeless.train']['degree_1'])))

    print('train loaders', len(train_loaders))
    return TrainDataLoader(train_loaders, support=support, query=query, pad_idx=pad_idx)


def get_test_loader(full_data, support, query, pad_idx):
    loader = []
    for filename in full_data:
        # support
        support_data = full_data[filename]['degree_0']['support_data'][0:support] + full_data[filename]['degree_1']['support_data'][0:support]\
            + full_data[filename]['degree_2']['support_data'][0:support] + full_data[filename]['degree_3']['support_data'][0:support]\
                + full_data[filename]['degree_4']['support_data'][0:support]
        support_data = batch_padding(support_data, pad_idx)

        support_target = full_data[filename]['degree_0']['support_target'][0:support] + full_data[filename]['degree_1']['support_target'][0:support]\
            + full_data[filename]['degree_2']['support_target'][0:support] + full_data[filename]['degree_3']['support_target'][0:support]\
                + full_data[filename]['degree_4']['support_target'][0:support]
        support_target = torch.tensor(support_target)

        # query
        degree_0_dl = DataLoader(Dataset(full_data[filename]['degree_0'], pad_idx), batch_size=query , shuffle=False, drop_last=False, **kwargs)
        degree_1_dl = DataLoader(Dataset(full_data[filename]['degree_1'], pad_idx), batch_size=query , shuffle=False, drop_last=False, **kwargs)
        degree_2_dl = DataLoader(Dataset(full_data[filename]['degree_2'], pad_idx), batch_size=query , shuffle=False, drop_last=False, **kwargs)
        degree_3_dl = DataLoader(Dataset(full_data[filename]['degree_3'], pad_idx), batch_size=query , shuffle=False, drop_last=False, **kwargs)
        degree_4_dl = DataLoader(Dataset(full_data[filename]['degree_4'], pad_idx), batch_size=query , shuffle=False, drop_last=False, **kwargs)
        
        # combine
        for dl in [degree_0_dl, degree_1_dl, degree_2_dl, degree_3_dl, degree_4_dl]:
            for batch_data, batch_target in dl:
                support_data_cp, support_target_cp = copy.deepcopy(support_data), copy.deepcopy(support_target)
                support_data_cp, batch_data = padding_2(support_data_cp, batch_data, pad_idx)
                data = torch.cat([support_data_cp, batch_data], dim=0)
                target = torch.cat([support_target_cp, batch_target], dim=0)

                # print("length of support_data, length of batch data = ", len(support_data), len(batch_data))
                # print("length of support target, length of target", len(support_target), len(target))
                loader.append((data, target))
    print('test loader length', len(loader))
    return loader


def main():
    train_domains, test_domains = get_domains(data_path, config['data']['filtered_list'], config['data']['target_list'])

    train_data = get_train_data(data_path, train_domains)
    # print(train_data['homeless.train']['degree_1']['data'][0])
    # print(train_data['homeless.train']['degree_1']['target'][0])

    dev_data, test_data = get_test_data(data_path, test_domains)
    # print(len(dev_data['disabled.dev']['degree_1']['support_data']))
    # print(test_data['disabled.test']['degree_1']['support_target'])

    vocabulary = get_vocabulary(train_data, min_freq=int(config['data']['min_freq']))
    pad_idx = vocabulary.padding_idx
    pickle.dump(vocabulary, open(os.path.join(config['data']['path'], config['data']['vocabulary']), 'wb'))

    train_data = idx_all_data(train_data, vocabulary)
    dev_data = idx_all_data(dev_data, vocabulary)
    test_data = idx_all_data(test_data, vocabulary)
    # print(dev_data['disabled.dev']['degree_1']['support_data'])
    # print(dev_data['disabled.dev']['degree_1']['support_target'])

    support = int(config['model']['support'])
    query = int(config['model']['query'])
    train_loader = get_train_loader(train_data, support, query, pad_idx)
    dev_loader = get_test_loader(dev_data, support, query, pad_idx)
    test_loader = get_test_loader(test_data, support, query, pad_idx)
    
    pickle.dump(train_loader, open(os.path.join(config['data']['path'], config['data']['train_loader']), 'wb'))
    pickle.dump(dev_loader, open(os.path.join(config['data']['path'], config['data']['dev_loader']), 'wb'))
    pickle.dump(test_loader, open(os.path.join(config['data']['path'], config['data']['test_loader']), 'wb'))


if __name__ == "__main__":
    # config
    config = configparser.ConfigParser()

    config.read("config.ini")

    # seed
    seed = int(config['data']['seed'])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    data_path = config['data']['path']
    main()
