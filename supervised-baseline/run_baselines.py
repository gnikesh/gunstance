import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
import torch
import json
import argparse
import pathlib
import copy

from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers import AutoModelForSequenceClassification
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import transformers
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import matplotlib.pyplot as plt
# from calibration_error import get_bucket_scores, get_bucket_confidence, get_bucket_accuracy, \
#                 create_one_hot, cross_entropy, get_best_temperature, calculate_calibration_error, process_output_dicts
from calibration_error import calculate_calibration_error

from baseline_models import get_PGCNN, get_GCAE, get_Kim_CNN, get_BiLSTM, get_TAN, get_ATGRU
from dataset_utils import get_dataset_tensors
from calibration_error import calculate_calibration_error


BASE_DIR = '/home/g/gnikesh/projects/shooting/shooting-project/DeCoTa-Text'
MAX_LEN = 128


def evaluate_with_ece(clf_model, data_loader, loss_function, epoch, batch_size, temperature=1.0):
    clf_model.eval()
    output_dicts = []
    total_loss = []
    with torch.no_grad():
        for i, items in enumerate(data_loader):
            data, target, length, t_word, tokens = items
            
            data, t_word = data.float(), t_word.float()
            # data, target, length, t_word, tokens = data.cuda(), target.cuda(), length.cuda(), t_word.cuda(), tokens.cuda()
            data, target, length, t_word, tokens = data.to(device), target.to(device), length, t_word.to(device), tokens.to(device)
            output = model(data, length, epoch, t_word, tokens)
            logits = output.clone().detach()
            
            loss = loss_function(output, target)
            total_loss.append(loss.item())
            
            for j in range(logits.size(0)):
                probs = F.softmax(logits[j], -1)
                output_dict = {
                    'index': batch_size * i + j,
                    'true': target[j].item(),
                    'pred': logits[j].argmax().item(),
                    'conf': probs.max().item(),
                    'logits': logits[j].cpu().numpy().tolist(),
                    'probs': probs.cpu().numpy().tolist(),
                }
                output_dicts.append(output_dict)
    
    result_dict = calculate_calibration_error(output_dicts, 3, temperature=temperature, average='weighted')
    result_dict['loss'] = np.mean(total_loss)
    return result_dict



def train_model(model, train_tensor, test_tensor, valid_tensor, checkpoint_path, args, device):
    batch_size = args.batch_size

    train_loader = DataLoader(train_tensor, shuffle=True, batch_size=batch_size)

    # y_valid = valid_tensor[:][1]
    valid_loader = DataLoader(valid_tensor, shuffle=False, batch_size=batch_size)

    # y_test = test_tensor[:][1]
    test_loader = DataLoader(test_tensor, shuffle=False, batch_size=batch_size)

    learning_rate = args.lr

    model.to(device)

    loss_function = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)


    # Train
    the_last_f1 = 0.0
    best_eval_f1 = 0.0
    trigger_times = 0
    best_state_dict = None
    best_epoch = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, epoch_train_accu = [], []
        for data, target, length, t_word, tokens in train_loader:
            data, t_word, = data.float(), t_word.float()
            # data, target, length, t_word, tokens = data.cuda(), target.cuda(), length.cuda(), t_word.cuda(), tokens.cuda()
            data, target, length, t_word, tokens = data.to(device), target.to(device), length, t_word.to(device), tokens.to(device)
            optimizer.zero_grad()

            output1 = model(data, length, epoch, t_word, tokens)
            
            loss = loss_function(output1, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 4) # type: ignore
            optimizer.step()
            
            preds = torch.argmax(F.softmax(output1, dim=-1), dim=-1)

            epoch_loss.append(loss.item())
            epoch_train_accu.append(accuracy_score(target.tolist(), preds.tolist()))

        # Evaluate Part
        val_dict = evaluate_with_ece(model, valid_loader, loss_function, epoch, args.batch_size, temperature=1.0)
        val_accu, val_pr, val_rc, val_f1, val_ece = val_dict['accuracy'], val_dict['precision'], val_dict['recall'], val_dict['f1-score'], val_dict['expected error']

        result_string = (f"{epoch}/{args.epochs} "
                        f"Loss: {np.mean(epoch_loss):.4f} "
                        f"Train accu: {np.mean(epoch_train_accu):.4f} "
                        f"Val accu: {val_accu:.4f} "
                        f"Val f1: {val_f1:.4f} "
                        f"Val loss: {val_dict['loss']} "
                        f"Val ece: {val_ece:.4f}")
        print(result_string)
        
        # For early stopping
        if val_f1 < the_last_f1:
        # if epoch_eval_loss > the_last_loss:
            trigger_times += 1
            if trigger_times >= args.patience: # type: ignore
                print("Early stopping the training with patience: {}".format(args.patience)) # type: ignore
                break
        else:
            trigger_times = 0
        the_last_f1 = val_f1

        # To save the best model
        if val_f1 > best_eval_f1:
            best_eval_f1 = val_f1
            best_state_dict = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            # torch.save({
            #     'epoch': best_epoch, 
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     'loss': val_dict['loss']}, os.path.join(checkpoint_path, 'best_model.pt'))

    # Load best model
    # checkpoint = torch.load(os.path.join(checkpoint_path, 'best_model.pt'))    
    # model.load_state_dict(checkpoint['model_state_dict'])
    # epoch = checkpoint['epoch']
    # loss = checkpoint['loss']
    model.load_state_dict(best_state_dict)
    
    # Saving model
    torch.save({
        'epoch': best_epoch,
        'model_state_dict': model.state_dict()
    }, os.path.join(checkpoint_path, 'best_model.pt'))

    # Calculate the best temperature
    val_dict = evaluate_with_ece(model, valid_loader, loss_function, best_epoch, args.batch_size, temperature=None)

    best_temperature = val_dict['temperature']
    # Test part after end of training epochs
    test_dict = evaluate_with_ece(model, test_loader, loss_function, best_epoch, args.batch_size, temperature=best_temperature)

    return test_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=120, help="Total epoch to train the model")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training")
    parser.add_argument('--patience', type=int, default=10, help="Number of epochs to wait until early stopping")
    parser.add_argument('--lr', type=float, default=1e-5, help="Learning rate for training the model")
    parser.add_argument('--weight_decay', type=float, default=4e-5, help="Learning rate for training the model")
    parser.add_argument('--gpu', type=int, default=0, help="GPU number to use")
    parser.add_argument('--train_csv', type=str, default='ban_all7/train.csv', help='train csv file with event/train.csv')
    parser.add_argument('--test_csv', type=str, default='ban_all7/test.csv', help='test csv file with event/test.csv')
    parser.add_argument('--valid_csv', type=str, default='ban_all7/valid.csv', help='valid csv file with event/valid.csv')
    parser.add_argument('--model_name', type=str, choices=(['PGCNN', 'GCAE', 'Kim_CNN', 'BiLSTM', 'TAN', 'ATGRU']), 
                        default='PGCNN', help='baseline model to run')
    
    args = parser.parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # dataset_path = '/home/g/gnikesh/projects/shooting/shooting-project/DeCoTa-Text/data/shootings_splits_final/AUM-ST-Datasets-Final'
    dataset_path = '/home/g/gnikesh/projects/shooting/shooting-project/DeCoTa-Text/data/shootings_splits_camera_ready/AUM-ST-Datasets-CameraReady'

    results_path = os.path.join(pathlib.Path(__file__).parent, 'results_camera_ready')

    # train_df, test_df, valid_df = get_dataset(dataset_path, args.train_csv, args.test_csv, args.valid_csv)
    train_tensor, test_tensor, valid_tensor = get_dataset_tensors(dataset_path, 
                                                                  args.train_csv, 
                                                                  args.test_csv,
                                                                  args.valid_csv, 
                                                                  MAX_LEN=128, 
                                                                  model_name="digitalepidemiologylab/covid-twitter-bert")


    models = [get_PGCNN(), get_GCAE(), get_Kim_CNN(), get_BiLSTM(), get_TAN(), get_ATGRU()]
    # models = [get_ATGRU()]
    
    run_name = args.train_csv.split('/')[0] + '_' + args.test_csv.split('/')[0]

    runs_list = [1, 2, 3, 4, 5]
    # runs_list = [2]

    for run_id in runs_list:
        for model in models:
            model_name = model.model_name
            print('*' * 40)
            print(f'run_id: {run_id}, training model: {model_name}')
            print('*' * 40)

            checkpoint_path = os.path.join(results_path, model_name, run_name, f'run_id_{run_id}', 'saved_model')
            os.makedirs(checkpoint_path, exist_ok=True)

            # Final results on the test set with temperature scaled model
            results_dict = train_model(model, train_tensor, test_tensor, valid_tensor, checkpoint_path, args, device)
            print(results_dict)
            with open(os.path.join(results_path, model_name, run_name, f'run_id_{run_id}', f'test_results_run_id_{run_id}.json'), 'w') as fwrite:
                json.dump(results_dict, fwrite)

