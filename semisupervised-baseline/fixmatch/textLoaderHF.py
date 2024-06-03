import os
from collections import Counter
import pandas as pd
import torch
from torch.utils.data import DataLoader
from textDatasetHF import TextDatasetHF
from transformers import AutoTokenizer


def get_labels_and_frequencies(path, label_column='Label'):
    df = pd.read_csv(path)
    label_freqs = Counter(df[label_column])
    print(f'Labels frequencies: {label_freqs}')

    return list(sorted(label_freqs.keys())), label_freqs


def get_datasets_v2(args):    
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model)

    args.labels, args.label_freqs = get_labels_and_frequencies(
        os.path.join(args.data_path, args.task, f"{args.train_file}.csv"), args.label_column
    )
    label_weights = [args.label_freqs[x] for x in args.labels]
    args.label_weights = [max(label_weights) / x for x in label_weights]
    print(f'Labels: {args.labels}')
    print(f'Weights: {args.label_weights}')
    args.n_classes = len(args.labels)

    labeled_dataset = TextDatasetHF(
        os.path.join(args.data_path, args.task, f"{args.train_file}.csv"),
        args.augmentations_file,
        tokenizer,
        args,
        text_aug1=args.text_soft_aug,
        text_aug2='none'
    )

    args.train_data_len = len(labeled_dataset)

    unlabeled_dataset = TextDatasetHF(
        os.path.join(args.data_path, args.eval_task, f"{args.unlabeled_dataset}.csv"),
        args.augmentations_file,
        tokenizer,
        args,
        text_aug1=args.text_soft_aug,
        text_aug2=args.text_hard_aug,
        min_num_examples=args.train_data_len*args.mu
    )

    dev_dataset = TextDatasetHF(
        os.path.join(args.data_path, args.task, f"{args.valid_file}.csv"),
        args.augmentations_file,
        tokenizer,
        args,
        text_aug1='none',
        text_aug2='none'
    )

    test_dataset = TextDatasetHF(
        os.path.join(args.data_path, args.eval_task, f"{args.test_file}.csv"),
        args.augmentations_file,
        tokenizer,
        args,
        text_aug1='none',
        text_aug2='none'
    )
    
    return labeled_dataset, unlabeled_dataset, dev_dataset, test_dataset


def get_data_loaders(args):
    labeled_dataset, unlabeled_dataset, dev_dataset, test_dataset = get_datasets_v2(args)
    
    if args.local_rank == 0:
        torch.distributed.barrier()

    labeled_loader = DataLoader(
        labeled_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_workers,
        drop_last=True
    )

    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        shuffle=True,
        batch_size=args.batch_size*args.mu,
        num_workers=args.n_workers,
        drop_last=True
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size, # args.batch_size*(args.mu+1)
        shuffle=False,
        num_workers=args.n_workers,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_workers,
    )

    return labeled_loader, unlabeled_loader, dev_loader, test_loader
