import random
import torch
from torch.utils.data import Dataset
import pandas as pd


def filter_columns(columns, aug_method='BT', min_strength=None, max_strength=None):
    columns = [col for col in columns if aug_method in col]
    if not columns:
        return []

    strengths = set([int(col.split('_')[1]) for col in columns])
    min_strength = min_strength or min(strengths)
    max_strength = max_strength or max(strengths)

    results = []
    for strength in range(min_strength, max_strength+1):
        results += [col for col in columns if int(col.split('_')[1]) == strength]
    return results


class TextDatasetHF(Dataset):

    def __init__(self, csv_path, aug_path, tokenizer, args, text_aug1='none', text_aug2='none', min_num_examples=None):
        df = pd.read_csv(csv_path)

        def _text_aug_to_aug_columns(text_aug):
            if '~' in text_aug:
                aug_method, min_strength, max_strength = text_aug.split('~')
                min_strength, max_strength = int(min_strength), int(max_strength)
                return aug_method, min_strength, max_strength
            else:
                return text_aug, None, None

        # aug_path should be the path to bt_augmentations.csv, aug_method shoul be "BT" for train, "NONE" for valid/test
        # Read and filter the columns containing augmentations with the desired strength, in order to save memory
        aug_columns = pd.read_csv(aug_path, nrows=0).columns.tolist()
        aug_method1, min_strength1, max_strength1 = _text_aug_to_aug_columns(text_aug1)
        aug_method2, min_strength2, max_strength2 = _text_aug_to_aug_columns(text_aug2)
        self.aug_columns1 = filter_columns(aug_columns, aug_method1, min_strength1, max_strength1)
        self.aug_columns2 = filter_columns(aug_columns, aug_method2, min_strength2, max_strength2)

        # Read augmentations dataframe and set text_column as index for fast lookup
        self.aug_df = pd.read_csv(aug_path, usecols=self.aug_columns1+self.aug_columns2+['Text'])
        self.aug_df.set_index('Text', inplace=True)
        print(f'{len(self.aug_columns1), len(self.aug_columns2)} augmentations available per sample')
                
        if min_num_examples is not None:
            mul_factor = min_num_examples // df.shape[0] + 1
            print(f'Multiplication factor {mul_factor} (required {min_num_examples} / existing {df.shape[0]})')
            df = pd.concat([df]*mul_factor, ignore_index=True)

        self.data = df

        self.num_classes = self.data[args.label_column].unique().shape[0]
        print(f'Loaded {self.data.shape[0]} examples distributed from {self.num_classes} classes.')

        self.args = args
        self.tokenizer = tokenizer
        self.max_seq_len = args.max_seq_len
        self.text_aug1 = text_aug1
        self.text_aug2 = text_aug2

    def __len__(self):
        return self.data.shape[0]  # len(self.data)

    def _get_inputs_dict(self, text, label, keys_prefix, query):
        texts = [text, query] if query else text  # I think it would make more sense to use query first
        inputs = self.tokenizer.encode_plus(
            texts,
            None,
            add_special_tokens=True,
            max_length=self.max_seq_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )

        return {
            f'{keys_prefix}ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            f'{keys_prefix}masks': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            f'{keys_prefix}token_type_ids': torch.tensor(inputs["token_type_ids"], dtype=torch.long),
            f'{keys_prefix}targets': torch.tensor(label, dtype=torch.long)
        }

    def _get_augmented_sentence_v2(self, row, aug_columns, keys_prefix=''):
        text = str(row[self.args.text_column])
        query = str(row[self.args.query_column]) if self.args.query_column else None

        if aug_columns:
            aug_col = random.choice(aug_columns)
            text = self.aug_df.loc[text, aug_col]

        label_string = row[self.args.label_column]
        label_int = self.args.labels.index(label_string) if label_string in self.args.labels else -1

        return self._get_inputs_dict(text, label_int, keys_prefix, query)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        inputs1 = self._get_augmented_sentence_v2(row, self.aug_columns1, 'weak_')
        inputs2 = self._get_augmented_sentence_v2(row, self.aug_columns2, 'strong_')

        inputs = inputs1 | inputs2
        return inputs
