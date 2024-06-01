import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from distutils.command.config import config
from sklearn.utils import shuffle

import argparse
import logging
import numpy as np
import pathlib
import pandas as pd
import random
import sys
import transformers

from absl import app
from absl import flags

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from collections import defaultdict
from copy import deepcopy
from scipy.special import softmax
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from transformers import *
from tqdm import tqdm
from train import evaluate

from aum import AUMCalculator

import logging
import math
import numpy as np
import os
import torch

from data import TestDataset, TrainDataset, TensorboardLog
from data import sample_split, get_eval_dataset, retrieve_augmentations, prepare_shooting_dataset
from utils import save_flags
from train import train_supervised, predict_unlabeled, train_ssl

import logging
import re


def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)


set_global_logging_level(logging.ERROR, ["transformers", "torch"])


FLAGS = flags.FLAGS
# flags.DEFINE_string("pt_teacher_checkpoint",
#                     "bert-base-uncased", "Initialization checkpoint.")

# Paths
flags.DEFINE_string("base_dir",
                    "/data/gnikesh/projects/shooting/shooting-project/AUM-ST/result_temp", "Path to the base dir where to save all the results")

flags.DEFINE_string("pt_teacher_checkpoint",
                    "cardiffnlp/twitter-roberta-base-sep2022", "Initialization checkpoint.")

flags.DEFINE_string('saved_teacher_model_path',
                    None, # 'path_to/saved_model',
                    'path to the saved supervised model for initial teacher')

flags.DEFINE_string("augmentation_dir",
                    "guns should be banned", "Directory that contains the output of the augmentation script.")

flags.DEFINE_string("dataset_path",
                     "/data/gnikesh/projects/shooting/shooting-project/DeCoTa-Text/data/shootings_splits_camera_ready", 'Main path to the dataset...')

flags.DEFINE_string("validation_path",
                    "ban_all7/valid.csv", "Path to the validation set csv file.")
flags.DEFINE_string("test_path", 
                    "ban_all7/test.csv", "Path to test set csv file.")
flags.DEFINE_string("train_path",
                    "ban_all7/train.csv", "Path to train set csv file.")

flags.DEFINE_string("unlabelled_path", 
                    "ban_all7/unlabeled.csv", "Path to unlabelled set csv file.")

flags.DEFINE_string("intermediate_model_path", 
                    "result_temp/saved_model", "Directory where to save intermediate models. Use different paths if using multiple parallel training jobs.")

flags.DEFINE_string("tensorboard_dir",
                    "result_temp/log", "Where to save stats about training incl. impurity, mask rate, loss, validation acc, etc.")
flags.DEFINE_string("aum_save_dir",
                    "result_temp/aum_dir", "Directory where AUM values will be saved")
flags.DEFINE_string("hyperparameters_save_dir", "result_temp", "Folder where to save the used hyperparameters")

flags.DEFINE_integer("num_labels", 10, "How many labels per class to use.")
flags.DEFINE_integer("max_seq_len", 128,
                     "Max sequence length. Use 512 for IMDB.")
flags.DEFINE_integer("num_classes", 3, "Number of classes in the dataset.")
flags.DEFINE_integer("seed", 1, "Seed for sampling datasets.")

flags.DEFINE_integer("weak_augmentation_min_strength", 0,
                     "Minimum strength of weak augmentations. Read instructions in the repository for a description.")
flags.DEFINE_integer("weak_augmentation_max_strength", 2,
                     "Maximum strength of weak augmentations. Read instructions in the repository for a description.")
flags.DEFINE_integer("strong_augmentation_min_strength", 3,
                     "Minimum strength of strong augmentations. Read instructions in the repository for a description.")
flags.DEFINE_integer("strong_augmentation_max_strength", 40,
                     "Maximum strength of strong augmentations. Read instructions in the repository for a description.")

flags.DEFINE_integer("sup_batch_size", 32, "Supervised batch size to use.")
flags.DEFINE_integer("unsup_batch_size", 128, "Unsupervised batch size to use.")
flags.DEFINE_integer("inference_batch_size", 256,
                     "Batch size to use for evaluation.")

# Use 5
flags.DEFINE_integer("initial_num", 
                     5, "Number of initial supervised models to train.")

# Default: 10 for both
flags.DEFINE_integer("supervised_patience", 10,
                     "Patience for fully supervised model.")
flags.DEFINE_integer("unsupervised_patience", 10,
                     "Patience for AUM-ST SSL training.")

# Default: 100
flags.DEFINE_integer("self_training_steps", 100,
                     "Number of self-training epochs.")

# Default: 15
flags.DEFINE_integer("unlabeled_epochs_per_step", 20,
                     "Number of epochs to use in each self-training step.")

# Default: 20
flags.DEFINE_integer("supervised_once_epochs", 10,
                     "Number of epochs to use in supervised once epochs.")

# use 0.7
flags.DEFINE_float("threshold", 
                   0.7, "Threshold for pseudo-labeling unlabeled data.")

# aum_percentile .90
flags.DEFINE_float("aum_percentile", 0.50, "Aum percentile.")

flags.DEFINE_string("experiment_id", "1",
                    "Name of the experiment. Will be used for tensorboard.")
flags.DEFINE_integer("gpu_id", 0, "GPU Id to use")



def get_ids_labels(file_path):
    df = pd.read_csv(file_path)
    ids = list(df['Id'])
    labels = list(df['Label'])
    query = list(df['Query'])
    
    labels_dict = {k: v for k, v in zip(ids, labels)}
    query_dict = {k: v for k, v in zip(ids, query)}

    return ids, labels_dict, query_dict


def main(argv):
    device = torch.device(f"cuda:{FLAGS.gpu_id}" if torch.cuda.is_available() else "cpu")

    save_flags(FLAGS)
    tokenizer = AutoTokenizer.from_pretrained(
        FLAGS.pt_teacher_checkpoint, verbose=False)
    available_augmentations = os.listdir(os.path.join(FLAGS.dataset_path, 'AUM-ST-Augmentations-Final', FLAGS.augmentation_dir))
    
    # Although in a real-world setup we don't have access to unlabeled labels, we keep
    # track of them here to compute various metrics such as impurity or mask rate.
    # ids_train, labels_train, ids_unlabeled, labels_unlabeled = sample_split(
    #     available_augmentations[0], FLAGS)

    ids_train, labels_train, queries_train = get_ids_labels(os.path.join(FLAGS.dataset_path, 'AUM-ST-Datasets-CameraReady', FLAGS.train_path))
    ids_unlabeled, labels_unlabeled, queries_unlabeled = get_ids_labels(os.path.join(FLAGS.dataset_path, 'AUM-ST-Datasets-CameraReady', FLAGS.unlabelled_path))
    
    validation_dataset = get_eval_dataset(
        os.path.join(FLAGS.dataset_path, 'AUM-ST-Datasets-CameraReady', FLAGS.validation_path), 
        tokenizer, sampling_strategy=-1)
    
    test_dataset = get_eval_dataset(os.path.join(FLAGS.dataset_path, 'AUM-ST-Datasets-CameraReady', FLAGS.test_path), 
                                    tokenizer)

    weak_train_dict, weak_unlabeled_dict, strong_unlabeled_dict = retrieve_augmentations(
        available_augmentations, FLAGS.weak_augmentation_min_strength, FLAGS.weak_augmentation_max_strength, 
        FLAGS.strong_augmentation_min_strength, FLAGS.strong_augmentation_max_strength, 
        ids_train, ids_unlabeled, os.path.join(FLAGS.dataset_path, 'AUM-ST-Augmentations-Final', FLAGS.augmentation_dir))
    
    print(set(ids_train).difference(set(weak_train_dict.keys())))
    
    train_dataset = TrainDataset(
        weak_train_dict, labels_train, ids_train, tokenizer)
    
    weakly_augmented_unlabeled_dataset = TrainDataset(
        weak_unlabeled_dict, labels_unlabeled, ids_unlabeled, tokenizer)

    labeled_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=FLAGS.sup_batch_size, shuffle=True)
    
    unlabeled_weak_dataloader = torch.utils.data.DataLoader(
        weakly_augmented_unlabeled_dataset, batch_size=FLAGS.unsup_batch_size, shuffle=True)
    
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=FLAGS.inference_batch_size, shuffle=False)
    
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=FLAGS.inference_batch_size, shuffle=False)

    tensorboard_logger = TensorboardLog(os.path.join(FLAGS.base_dir, 
        FLAGS.tensorboard_dir), FLAGS.experiment_id)

    best_f1_overall = 0
    print('initial supervised training...')
    best_f1_overall, supervised_validation_f1, supervised_validation_loss, supervised_test_f1, supervised_test_loss = train_supervised(
        FLAGS, best_f1_overall, labeled_dataloader, validation_dataloader, test_dataloader)
    
    best_temperature = 1.0
    print('self training...')
    for self_training_epoch in range(FLAGS.self_training_steps):

        tensorboard_updater_dict = {}
        print('running predict unlabeled...')
        # Make predictions on weakly-augmented unlabeled examples.
        ind, total, num_correct, pseudo_labels_dict = predict_unlabeled(
            FLAGS, unlabeled_weak_dataloader)

        if len(ind) == 0:
            print('-----------------------------------------')
            print('Nothing passes the threshold, Retraining supervised... Consider lowering the threshold if this message persists.')
            print('-----------------------------------------')
            best_f1_overall = 0
            best_f1_overall, supervised_validation_f1, supervised_validation_loss, supervised_test_f1, supervised_test_loss = train_supervised(
                FLAGS, best_f1_overall, labeled_dataloader, validation_dataloader, test_dataloader)
            continue

        # Update the log with statistics about the unlabeled data.
        tensorboard_updater_dict['mask_rate'] = 1 - float(len(ind)) / float(total)
        tensorboard_updater_dict['impurity'] = 1 - num_correct / float(len(ind))

        # Eliminate low-aum examples.
        assert pseudo_labels_dict != None
        threshold_pseudo_labels_dict = deepcopy(pseudo_labels_dict)
        threshold_examples = set(random.sample(list(pseudo_labels_dict.keys()),
                                            min(len(pseudo_labels_dict) / FLAGS.num_classes,
                                            len(pseudo_labels_dict) // 4)))
        
        print('-' * 40)
        print(f'total: {total} num_correct: {num_correct} pseudo_labels_dict_count: {len(pseudo_labels_dict)}')
        print('Selected', len(threshold_examples), 'threshold examples.')
        print('-' * 40)

        for e in threshold_examples:
            threshold_pseudo_labels_dict[e] = FLAGS.num_classes
        
        new_thrs = set()
        for elem in threshold_examples:
            new_thrs.add(str(elem))
        
        aum_calculator = AUMCalculator(os.path.join(FLAGS.base_dir, FLAGS.aum_save_dir), compressed=False)
        train_ssl(FLAGS, best_f1_overall, train_dataset, strong_unlabeled_dict, labels_unlabeled, ind, tokenizer,
                  threshold_pseudo_labels_dict, validation_dataloader, test_dataloader, 4, use_aum=True, aum_calculator=aum_calculator)
        aum_calculator.finalize()

        aum_values_df = pd.read_csv(os.path.join(FLAGS.base_dir,
            FLAGS.aum_save_dir, 'aum_values.csv'))
        threshold_examples_aum_values = []
        non_threshold = []
        
        aum_values_df['sample_id'] = aum_values_df['sample_id'].astype(str)

        for i, row in aum_values_df.iterrows():
            if str(row['sample_id']) in new_thrs:
                threshold_examples_aum_values.append(float(row['aum']))
            else:
                non_threshold.append((str(row['sample_id']), float(row['aum'])))
        
        print('sizes: ', len(threshold_examples_aum_values), len(threshold_examples))
        # assert len(threshold_examples_aum_values) == len(threshold_examples)

        threshold_examples_aum_values.sort()

        id = int(float(len(threshold_examples_aum_values)) * (1 - FLAGS.aum_percentile))

        aum_value = threshold_examples_aum_values[id]

        print('-------------------------')
        print('AUM threshold', aum_value)
        print('-------------------------')
        filtered_ids = [tpl[0] for tpl in non_threshold if float(tpl[1]) > float(aum_value)]
        print('LEN DE FILTERED IDS', len(filtered_ids))

        resulting_dict = {}
        num_before_elimination = len(pseudo_labels_dict)
        print('-------------------------')
        print('Size of unlabeled set before AUM filtering:', num_before_elimination)
        print('-------------------------')
        for k in pseudo_labels_dict:
            if str(k) in filtered_ids or str(k) in new_thrs:
                resulting_dict[k] = pseudo_labels_dict[k]
        pseudo_labels_dict = resulting_dict
        num_after_elimination = len(pseudo_labels_dict)
        print('-------------------------')
        print('Size of unlabeled set after AUM filtering:', num_after_elimination)
        print('-------------------------')
        ind = list(pseudo_labels_dict.keys())

        print('train ssl with unlabeled sed after AUM filtering...')
        best_f1_overall, best_f1, corresponding_test, best_loss_validation, best_loss_test = train_ssl(
            FLAGS, best_f1_overall, train_dataset, strong_unlabeled_dict, labels_unlabeled, ind, tokenizer, pseudo_labels_dict, validation_dataloader, test_dataloader, FLAGS.unlabeled_epochs_per_step) # type: ignore

        tensorboard_updater_dict['ssl/validation_f1'] = best_f1
        tensorboard_updater_dict['ssl/test_f1'] = corresponding_test
        tensorboard_updater_dict['ssl/validation_loss'] = best_loss_validation
        tensorboard_updater_dict['ssl/test_loss'] = best_loss_test
        tensorboard_updater_dict['ssl/validation_best_f1_overall'] = best_f1_overall

        tensorboard_updater_dict['ssl/aum_threshold'] = aum_value
        tensorboard_updater_dict['ssl/aum_eliminated'] = num_before_elimination - num_after_elimination

        tensorboard_updater_dict['supervised/validation_f1'] = supervised_validation_f1
        tensorboard_updater_dict['supervised/validation_loss'] = supervised_validation_loss
        tensorboard_updater_dict['supervised/test_f1'] = supervised_test_f1
        tensorboard_updater_dict['supervised/test_loss'] = supervised_test_loss

        tensorboard_logger.update(
            tensorboard_updater_dict, self_training_epoch)

    model = AutoModelForSequenceClassification.from_pretrained(os.path.join(FLAGS.base_dir,
        FLAGS.intermediate_model_path))
    
    model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    f1_macro_test, _, ece_dict = evaluate(
        model, test_dataloader, loss_fn, FLAGS.inference_batch_size, get_ece=True, temperature=best_temperature, FLAGS=FLAGS)
    
    print('Final test f1:', f1_macro_test)
    print("ECE DICT: ", ece_dict)
    
    with open(os.path.join(FLAGS.base_dir, FLAGS.hyperparameters_save_dir, 'final_results.json'), 'w') as fwriter:
        fwriter.write(str(ece_dict))


if __name__ == "__main__":
    app.run(main)

# Note:
# Replace model and also update f1