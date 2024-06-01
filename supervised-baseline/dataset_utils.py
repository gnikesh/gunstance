import numpy as np
import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import TensorDataset, DataLoader

def tokenize_string(s: str, tokenizer, max_length=512):
    if max_length > 512:
        max_length = 512
        print('WARNING: \'max_lenth\' of function tokenize_string cannot exceed 512. Thus \'max_lenth\' has been set to 512.')
    tokenized_string = torch.from_numpy(np.array(tokenizer.encode(s)))
    natural_token_length = len(tokenized_string)
    if natural_token_length > max_length:
        print('Notice: Truncating tweet {} -> {}.'.format(natural_token_length, max_length))
        temp = tokenized_string.tolist()
        temp = temp[:max_length]
        temp[-1] = 102
        tokenized_string = torch.tensor(temp)
    elif len(tokenized_string) < max_length:
        temp = tokenized_string.tolist()
        for x in range(max_length - len(tokenized_string)):
            temp.append(tokenizer.unk_token_id)
        tokenized_string = torch.tensor(temp)
    tokenized_string = tokenized_string.unsqueeze(0)
    return (tokenized_string.clone().detach(), min(natural_token_length, max_length))


def embed_string(s: str, tokenizer, bert_model, max_length=512):
    tokenized_string = tokenize_string(s, tokenizer, max_length)[0]
    embedded_string = bert_model(tokenized_string)
    s = embedded_string[0].clone().detach()
    return s


def get_dataset_tensors(dataset_path, train_csv, test_csv, valid_csv, MAX_LEN=128, model_name="digitalepidemiologylab/covid-twitter-bert"):
    def get_dataset(dataset_path, train_csv, test_csv, valid_csv):
        train_df = pd.read_csv(os.path.join(dataset_path, train_csv))
        test_df = pd.read_csv(os.path.join(dataset_path, test_csv))
        valid_df = pd.read_csv(os.path.join(dataset_path, valid_csv))

        return train_df, test_df, valid_df

    train_df, test_df, valid_df = get_dataset(dataset_path, train_csv, test_csv, valid_csv)

    train_pth, test_pth, valid_pth = train_csv.split('/'), test_csv.split('/'), valid_csv.split('/')


    if os.path.exists(os.path.join(dataset_path, train_pth[0], train_pth[1].replace('.csv', '_baseline_tensor.dst'))):
        train_tensor = torch.load(os.path.join(dataset_path, train_pth[0], train_pth[1].replace('.csv', '_baseline_tensor.dst')))

    else:
        print('creating train tensors')
        bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
        bert_model = AutoModel.from_pretrained(model_name)

        x_train, y_train, x_train_target = train_df['Text'], train_df['Label'], train_df['Query']

        # get sequence length for each sentence
        x_train_tokens = [tokenize_string(xi, bert_tokenizer, MAX_LEN) for xi in x_train]

        # Get lengths of tokens before padding
        x_train_len = np.array([xi[1] for xi in x_train_tokens])
        x_train_index = torch.stack([bert_model(tokens[0])[0].clone().detach() for tokens in x_train_tokens])
        
        train_target_index = torch.stack([embed_string(sentence, bert_tokenizer, bert_model, MAX_LEN) for sentence in x_train_target]).clone().detach() # type: ignore
        x_train = x_train_index.clone().detach()
        y_train = torch.tensor(y_train, dtype=torch.long).clone().detach()
        x_train_len = torch.tensor(x_train_len, dtype=torch.long).clone().detach()
        train_target_index = train_target_index.clone().detach()
        just_tokens = []

        for x in x_train_tokens:
            just_tokens.append(x[0])
            
        just_tokens = torch.stack(just_tokens).clone().detach()
        train_bert_tokens = just_tokens.clone().detach()
        train_tensor = TensorDataset(x_train, y_train, x_train_len, train_target_index, train_bert_tokens)

        torch.save(train_tensor, os.path.join(dataset_path, train_pth[0], train_pth[1].replace('.csv', '_baseline_tensor.dst')))


    if os.path.exists(os.path.join(dataset_path, test_pth[0], test_pth[1].replace('.csv', '_baseline_tensor.dst'))):
        test_tensor = torch.load(os.path.join(dataset_path, test_pth[0], test_pth[1].replace('.csv', '_baseline_tensor.dst')))

    else:
        print('creating test tensors')
        bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
        bert_model = AutoModel.from_pretrained(model_name)

        x_test, y_test, x_test_target = test_df['Text'], test_df['Label'], test_df['Query']
        
        # get sequence length for each sentence
        x_test_tokens = [tokenize_string(xi, bert_tokenizer, MAX_LEN) for xi in x_test]

        # Get lengths of tokens before padding
        x_test_len = np.array([xi[1] for xi in x_test_tokens])
        x_test_index = torch.stack([bert_model(tokens[0])[0].clone().detach() for tokens in x_test_tokens])


        test_target_index = torch.stack([embed_string(sentence, bert_tokenizer, bert_model, MAX_LEN) for sentence in x_test_target]).clone().detach() # type: ignore
        x_test = x_test_index.clone().detach()
        y_test = torch.tensor(y_test, dtype=torch.long).clone().detach()
        x_test_len = torch.tensor(x_test_len, dtype=torch.long).clone().detach()
        test_target_index = test_target_index.clone().detach()
        just_tokens = []

        for x in x_test_tokens:
            just_tokens.append(x[0])
        just_tokens = torch.stack(just_tokens).clone().detach()
        test_bert_tokens = just_tokens.clone().detach()
        test_tensor = TensorDataset(x_test, y_test, x_test_len, test_target_index, test_bert_tokens)

        torch.save(test_tensor, os.path.join(dataset_path, test_pth[0], test_pth[1].replace('.csv', '_baseline_tensor.dst')))

    
    if os.path.exists(os.path.join(dataset_path, valid_pth[0], valid_pth[1].replace('.csv', '_baseline_tensor.dst'))):
        valid_tensor = torch.load(os.path.join(dataset_path, valid_pth[0], valid_pth[1].replace('.csv', '_baseline_tensor.dst')))
    else:

        print('creating valid tensors')
        bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
        bert_model = AutoModel.from_pretrained(model_name)

        x_valid, y_valid, x_valid_target = valid_df['Text'], valid_df['Label'], valid_df['Query']

        # get sequence length for each sentence
        x_valid_tokens = [tokenize_string(xi, bert_tokenizer, MAX_LEN) for xi in x_valid]

        # Get lengths of tokens before padding
        x_valid_len = np.array([xi[1] for xi in x_valid_tokens])
        x_valid_index = torch.stack([bert_model(tokens[0])[0].clone().detach() for tokens in x_valid_tokens])

        valid_target_index = torch.stack([embed_string(sentence, bert_tokenizer, bert_model, MAX_LEN) for sentence in x_valid_target]).clone().detach() # type: ignore
        x_valid = x_valid_index.clone().detach()
        y_valid = torch.tensor(y_valid, dtype=torch.long).clone().detach()
        x_valid_len = torch.tensor(x_valid_len, dtype=torch.long).clone().detach()
        valid_target_index = valid_target_index.clone().detach()
        just_tokens = []

        for x in x_valid_tokens:
            just_tokens.append(x[0])
        just_tokens = torch.stack(just_tokens).clone().detach()
        valid_bert_tokens = just_tokens.clone().detach()
        valid_tensor = TensorDataset(x_valid, y_valid, x_valid_len, valid_target_index, valid_bert_tokens)

        torch.save(valid_tensor, os.path.join(dataset_path, valid_pth[0], valid_pth[1].replace('.csv', '_baseline_tensor.dst')))


    return train_tensor, test_tensor, valid_tensor


def get_dataset_tensors_BERT_NS(dataset_path, train_csv, test_csv, valid_csv, unlabeled_csv, MAX_LEN=128, model_name="digitalepidemiologylab/covid-twitter-bert"):
    def get_dataset(dataset_path, train_csv, test_csv, valid_csv, unlabeled_csv):
        train_df = pd.read_csv(os.path.join(dataset_path, train_csv))
        test_df = pd.read_csv(os.path.join(dataset_path, test_csv))
        valid_df = pd.read_csv(os.path.join(dataset_path, valid_csv))
        unlabeled_df = pd.read_csv(os.path.join(dataset_path, unlabeled_csv))

        return train_df, test_df, valid_df, unlabeled_df

    train_df, test_df, valid_df, unlabeled_df = get_dataset(dataset_path, train_csv, test_csv, valid_csv, unlabeled_csv)
    
    train_pth, test_pth, valid_pth, unlabeled_pth = train_csv.split('/'), test_csv.split('/'), valid_csv.split('/'), unlabeled_csv.split('/')



    if os.path.exists(os.path.join(dataset_path, train_pth[0], train_pth[1].replace('.csv', '_baseline_tensor_BERT-NS.dst'))):
        train_tensor = torch.load(os.path.join(dataset_path, train_pth[0], train_pth[1].replace('.csv', '_baseline_tensor_BERT-NS.dst')))

    else:
        print('creating train tensors')
        bert_tokenizer = AutoTokenizer.from_pretrained(model_name)

        x_train, y_train, x_train_target = train_df['Text'], train_df['Label'], train_df['Query']

        tokenized_x_train = bert_tokenizer(x_train, return_tensors='pt', padding=True, truncation=True, max_length=MAX_LEN)
        x_train_input_ids = tokenized_x_train['input_ids']
        x_train_attention_mask = tokenized_x_train['attention_mask']

        labels = torch.tensor(y_train, dtype=torch.long)
        train_tensor = TensorDataset(x_train_input_ids, x_train_attention_mask, labels)

        torch.save(train_tensor, os.path.join(dataset_path, train_pth[0], train_pth[1].replace('.csv', '_baseline_tensor_BERT-NS.dst')))


    if os.path.exists(os.path.join(dataset_path, test_pth[0], test_pth[1].replace('.csv', '_baseline_tensor_BERT-NS.dst'))):
        test_tensor = torch.load(os.path.join(dataset_path, test_pth[0], test_pth[1].replace('.csv', '_baseline_tensor_BERT-NS.dst')))

    else:
        print('creating test tensors')
        bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
        bert_model = AutoModel.from_pretrained(model_name)

        x_test, y_test, x_test_target = test_df['Text'], test_df['Label'], test_df['Query']
        tokenized_x_test = bert_tokenizer(x_test, return_tensors='pt', padding=True, truncation=True, max_length=MAX_LEN)
        x_test_input_ids = tokenized_x_test['input_ids']
        x_test_attention_mask = tokenized_x_test['attention_mask']
        
        labels = torch.tensor(y_test, dtype=torch.long)
        test_tensor = TensorDataset(x_test_input_ids, x_test_attention_mask, labels)

        torch.save(test_tensor, os.path.join(dataset_path, test_pth[0], test_pth[1].replace('.csv', '_baseline_tensor_BERT-NS.dst')))

    
    if os.path.exists(os.path.join(dataset_path, valid_pth[0], valid_pth[1].replace('.csv', '_baseline_tensor_BERT-NS.dst'))):
        valid_tensor = torch.load(os.path.join(dataset_path, valid_pth[0], valid_pth[1].replace('.csv', '_baseline_tensor_BERT-NS.dst')))
    else:

        print('creating valid tensors')
        bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
        bert_model = AutoModel.from_pretrained(model_name)

        x_valid, y_valid, x_valid_target = valid_df['Text'], valid_df['Label'], valid_df['Query']

        tokenized_x_val = bert_tokenizer(x_valid, return_tensors='pt', padding=True, truncation=True, max_length=MAX_LEN)
        x_val_input_ids = tokenized_x_val['input_ids']
        x_val_attention_mask = tokenized_x_val['attention_mask']

        labels = torch.tensor(y_valid, dtype=torch.long)
        valid_tensor = TensorDataset(x_val_input_ids, x_val_attention_mask, labels)
        torch.save(valid_tensor, os.path.join(dataset_path, valid_pth[0], valid_pth[1].replace('.csv', '_baseline_tensor_BERT-NS.dst')))


    if os.path.exists(os.path.join(dataset_path, unlabeled_pth[0], unlabeled_pth[1].replace('.csv', '_baseline_tensor_BERT-NS.dst'))):
        unlabeled_tensor = torch.load(os.path.join(dataset_path, unlabeled_pth[0], unlabeled_pth[1].replace('.csv', '_baseline_tensor_BERT-NS.dst')))

    else:
        print('creating valid tensors')
        bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
        bert_model = AutoModel.from_pretrained(model_name)

        x_unlabeled, y_unlabeled, x_unlabeled_target = valid_df['Text'], valid_df['Label'], valid_df['Query']

        tokenized_x_unlabeled = bert_tokenizer(x_unlabeled, return_tensors='pt', padding=True, truncation=True, max_length=MAX_LEN)
        x_unlabeled_input_ids = tokenized_x_unlabeled['input_ids']
        x_unlabeled_attention_mask = tokenized_x_unlabeled['attention_mask']

        # labels = torch.tensor(y_valid, dtype=torch.long)
        unlabeled_tensor = TensorDataset(x_unlabeled_input_ids, x_unlabeled_attention_mask)
        torch.save(unlabeled_tensor, os.path.join(dataset_path, valid_pth[0], valid_pth[1].replace('.csv', '_baseline_tensor_BERT-NS.dst')))


    return train_tensor, test_tensor, valid_tensor, unlabeled_tensor


