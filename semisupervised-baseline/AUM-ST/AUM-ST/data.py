import pandas as pd
import torch
import random
from collections import defaultdict
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import preprocessor as p

def clean_tweets(strings):
    # using lib to clean URL,hashtags...
    p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.RESERVED, p.OPT.SMILEY, p.OPT.MENTION)
    clean_data = p.clean(strings) 
    
    return clean_data


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, text_list, labels, tokenizer, seq_len=128):
        self.text_list = text_list
        self.labels = labels
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __getitem__(self, idx):
        tok = self.tokenizer.encode_plus(
            self.text_list[idx],
            padding='max_length', max_length=self.seq_len, truncation=True
        )
        # tok = self.tokenizer(
        #     [self.text_list[idx], self.test_target],
        #       padding='max_length', max_length=self.seq_len, truncation=True)

        item = {key: torch.tensor(tok[key]) for key in tok}
        item['lbl'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.text_list)


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, augmentation_dict, labels_dict, ids, tokenizer, pseudo_labels=None, seq_len=128):
        self.ids = ids
        self.augmentation_dict = augmentation_dict
        self.labels_dict = labels_dict
        self.tokenizer = tokenizer
        self.pseudo_labels = pseudo_labels
        self.seq_len = seq_len

    def __getitem__(self, idx):
        tok = self.tokenizer.encode_plus(
            random.choice(self.augmentation_dict[self.ids[idx]]),
            padding='max_length', max_length=self.seq_len, truncation=True
        )
        # tok = self.tokenizer(
        #     [random.choice(self.augmentation_dict[self.ids[idx]]), self.train_target],
        #     padding='max_length', max_length=self.seq_len, truncation=True)
        item = {key: torch.tensor(tok[key]) for key in tok}
        item['lbl'] = torch.tensor(
            self.labels_dict[self.ids[idx]], dtype=torch.long)
        if self.pseudo_labels != None:
            item['lbl'] = torch.tensor(
                self.pseudo_labels[self.ids[idx]], dtype=torch.long)
        item['idx'] = self.ids[idx]
        return item

    def __len__(self):
        return len(self.ids)


class TensorboardLog:
    def __init__(self, tb_dir, file_name):
        self.tb_dir = tb_dir
        self.writer = SummaryWriter(os.path.join(self.tb_dir, file_name))

    def update(self, tb_dict, it):
        for key, value in tb_dict.items():
            self.writer.add_scalar(key, value, it)


def sample_split(initial_set, FLAGS):
    random.seed(FLAGS.seed)
    label_frequencies_train = defaultdict(int)
    df = pd.read_csv(os.path.join(FLAGS.augmentation_dir, initial_set))
    for i, row in df.iterrows():
        label_frequencies_train[row['Label']] += 1

    train_label_to_set = {}
    for i, row in df.iterrows():
        if row['Label'] not in train_label_to_set:
            train_label_to_set[row['Label']] = set()
        train_label_to_set[row['Label']].add(row['Text'])

    training_set = set()
    unlabeled_set = set()

    for cls in train_label_to_set:
        crt_texts = train_label_to_set[cls]
        remaining = []
        if len(crt_texts) < FLAGS.num_labels:
            num_repeating = FLAGS.num_labels // len(crt_texts) + 1
            resulting_text = list(crt_texts) * num_repeating
            resulting_text = resulting_text[:FLAGS.num_labels]
        elif len(crt_texts) > FLAGS.num_labels:
            sampled = random.sample(crt_texts, FLAGS.num_labels)
            remaining = list(
                crt_texts.difference(set(sampled)))
            resulting_text = list(sampled)
        else:
            resulting_text = list(crt_texts)

        training_set = training_set.union(resulting_text)
        unlabeled_set = unlabeled_set.union(remaining)

    ids_train = []
    labels_train = {}
    ids_unlabeled = []
    labels_unlabeled = {}

    for i, row in df.iterrows():
        if row['Text'] in training_set:
            ids_train.append(row['Id'])
            labels_train[row['Id']] = row['Label']
        elif row['Text'] in unlabeled_set:
            ids_unlabeled.append(row['Id'])
            labels_unlabeled[row['Id']] = row['Label']

    return ids_train, labels_train, ids_unlabeled, labels_unlabeled


def get_eval_dataset(path, tokenizer, sampling_strategy=-1):

    df = pd.read_csv(path)
    text_list = list(df['Text'])
    labels_list = list(df['Label'])
    query_list = list(df['Query'])
    text_list = list(zip(text_list, query_list))

    tl, ll, ql = [], [], []
    if sampling_strategy != -1:
        zipped = list(zip(text_list, labels_list))
        random.shuffle(zipped)
        num_per_label = defaultdict(int)
        for elem in zipped:
            num_per_label[elem[1]] += 1
            if num_per_label[elem[1]] >= sampling_strategy:
                continue
            else:
                tl.append(elem[0])
                ll.append(elem[1])
        text_list = tl
        labels_list = ll

    dataset = TestDataset(text_list, labels_list, tokenizer)

    return dataset


def retrieve_simple(ids, paths, min_str, max_str):

    # # Nikesh ----
    def update_dict(dict1, dict2):
        updated_dict = dict1.copy()
        for key, value in dict2.items():
            if key in updated_dict:
                updated_dict[key].extend(value)
            else:
                updated_dict[key] = value

        return updated_dict

    id_set = set(ids)
    aug_dict = defaultdict(list)
    filtered_dfs = []
    for augmented_file in tqdm(paths):
        df = pd.read_csv(augmented_file)
        fdf = df[(df['Strength'] >= min_str) & (df['Strength'] <= max_str)]
        fdf = fdf[fdf['Id'].isin(id_set)]

        filtered_dfs.append(fdf)

    fdf = pd.concat(filtered_dfs, axis=0)
    # aug_dict = fdf[['Id', 'Text']].groupby(['Id'], group_keys=True)['Text'].apply(list).to_dict()
    
    grouped_data = df.groupby('Id').apply(lambda group: list(zip(group['Text'], group['Query']))).reset_index(name='GroupedValues')
    aug_dict = dict(zip(grouped_data['Id'], grouped_data['GroupedValues']))


    # # Nikesh -----

    # Original -----
    # id_set = set(ids)
    # aug_dict = defaultdict(list)
    # for augmented_file in tqdm(paths):
    #     df = pd.read_csv(augmented_file)
    #     if df['Strength'][0] >= min_str and df['Strength'][0] <= max_str:
    #         for i, row in df.iterrows():
    #             if row['Id'] in id_set:
    #                 aug_dict[row['Id']].append(row['Text'])

    # # Original -----
    
    return aug_dict


def retrieve_augmentations(available_augmentations, weak_aug_min, weak_aug_max, strong_aug_min, strong_aug_max, ids_train, ids_unlabeled, augmentation_dir):
    aug_paths = []

    for e in available_augmentations:
        aug_paths.append(os.path.join(augmentation_dir, e))

    print('Collecting weak augmentations for training set.')
    weak_train_dict = retrieve_simple(
        ids_train, aug_paths, weak_aug_min, weak_aug_max)
    print('Collecting weak augmentations for unlabeled set.')
    weak_unlabeled_dict = retrieve_simple(
        ids_unlabeled, aug_paths, weak_aug_min, weak_aug_max)
    print('Collecting strong augmentations for unlabeled set.')
    strong_unlabeled_dict = retrieve_simple(
        ids_unlabeled, aug_paths, strong_aug_min, strong_aug_max)

    return weak_train_dict, weak_unlabeled_dict, strong_unlabeled_dict


def prepare_shooting_dataset(topic, get_unlabelled=True):
    BASE_DIR = '/home/g/gnikesh/projects/shooting/shooting-project/DeCoTa-Text/data/shootings_splits_v2/'
    all_states = ['atlanta', 'boulder', 'buffalo', 'highlandpark', 'san', 'tulsa', 'uvalde']
    
    trains, tests, valids, unlabelleds = [], [], [], []
    for state in all_states:
        trains.append(pd.read_csv(os.path.join(BASE_DIR, f"{topic}_{state}", 'train.csv'), sep=','))
        tests.append(pd.read_csv(os.path.join(BASE_DIR, f"{topic}_{state}", 'test.csv')))
        valids.append(pd.read_csv(os.path.join(BASE_DIR, f"{topic}_{state}", 'valid.csv')))
        unlabelleds.append(pd.read_csv(os.path.join(BASE_DIR, f"{topic}_{state}", 'unlabeled.csv')))

    train_df = pd.concat(trains, axis=0)
    test_df = pd.concat(tests, axis=0)
    valid_df = pd.concat(valids, axis=0)
    unlabelled_df = pd.concat(unlabelleds, axis=0)
    
    train_df = train_df[['tweet_id', 'tweet', 'stance']]
    test_df = test_df[['tweet_id', 'tweet', 'stance']]
    valid_df = valid_df[['tweet_id', 'tweet', 'stance']]
    unlabelled_df = unlabelled_df[['tweet_id', 'tweet']]

    train_df['stance'].replace(['FAVOR','NONE','AGAINST', "NEUTRAL"], [1, 2, 0, 2], inplace=True)
    test_df['stance'].replace(['FAVOR','NONE','AGAINST', "NEUTRAL"], [1, 2, 0, 2], inplace=True)
    valid_df['stance'].replace(['FAVOR','NONE','AGAINST', "NEUTRAL"], [1, 2, 0, 2], inplace=True)
    unlabelled_df['stance'] = 1

    train_df.columns = ['Id', 'Text', 'Label']
    test_df.columns = ['Id', 'Text', 'Label']
    valid_df.columns = ['Id', 'Text', 'Label']
    unlabelled_df.columns = ['Id', 'Text', 'Label']

    save_path = '/home/g/gnikesh/projects/shooting/shooting-project/AUM-ST/data/labelled_data'
    train_df.to_csv(os.path.join(save_path, f'{topic}_train.csv'), sep=',', index=False)
    test_df.to_csv(os.path.join(save_path, f'{topic}_test.csv'), sep=',', index=False)
    valid_df.to_csv(os.path.join(save_path, f'{topic}_valid.csv'), sep=',', index=False)
    unlabelled_df.to_csv(os.path.join(save_path, f'{topic}_unlabelled.csv'), sep=',', index=False)
    
    return save_path
    


if __name__ == '__main__':
    prepare_shooting_dataset('regulate')
    
