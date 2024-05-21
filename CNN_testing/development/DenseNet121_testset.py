#!/usr/bin/env python3

import csv
import numpy as np
import optuna
import os
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.optim as optim
import torchvision.transforms as transforms
import wandb
from collections import defaultdict
from PIL import Image
from scipy import stats
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score,\
    roc_curve, auc, confusion_matrix
from torch.cuda.amp import autocast, GradScaler
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.models import densenet121, alexnet
import matplotlib.pyplot as plt


##########################---- SEED ----#######################################
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
###############################################################################

# Credentials
__author__ = "M.D.C. Jansen"
__version__ = "1.0"
__date__ = "19/02/2023"

# Parameter file path
param_path = r"D:/MDCJ/CNN-code/CNN-codes final runs/10X-CNN/DenseNet121/26-03-experiments/10X_test_parameters_DenseNet.csv"
models_path = r'D:/MDCJ/CNN-code/CNN-codes final runs/10X-CNN/DenseNet121/26-03-experiments/models'
input_modeldata = r"D:/MDCJ/CNN-code/CNN-codes final runs/10X-CNN/DenseNet121/FC-val/Best models/DenseNet121/Fold-5"

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, xlsx_path, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.df_labels = pd.read_excel(xlsx_path)
        self.all_images = [os.path.join(subdir, file)
                           for subdir, dirs, files in os.walk(self.root_dir)
                           for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        img_name = self.all_images[idx]
        img_file = img_name.split("CZ_")[-1]
        segments = img_name.split('_')
        if len(segments) <= 1:
            raise ValueError(f"Unexpected image name format for {img_name}")
        study_id = segments[1]
        label_entries = self.df_labels[self.df_labels['study_id'] == int(study_id)]['label'].values
        if len(label_entries) == 0:
            print(f"No label found for study_id: {study_id} in image {img_name}.")
            label = None
        else:
            label = label_entries[0]
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label, study_id, img_file


class CustomHead(nn.Module):
    def __init__(self, input_size, output_size, batch_norm):
        super(CustomHead, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.batch_norm = nn.BatchNorm1d(input_size) if batch_norm else None

    def forward(self, x):
        if self.batch_norm:
            x = self.batch_norm(x)
        x = self.fc(x)
        return x


def generate_filename(prefix, run_name, criterion, epoch):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        os.path.join("Best models", f"{prefix}_{run_name}_epoch_{epoch + 1}_{criterion}_batch_norm_{parameters['tl_bn']}.pt"))


def load_parameters(param_path):
    input_param = {}
    with open(param_path, 'r') as param_file:
        details = csv.DictReader(param_file)
        for d in details:
            input_param.setdefault(d['variable'], []).append(d['value'])
        param_file.close()
    input_variables = {
        'root_dir': ''.join(input_param['root_dir']),
        'xlsx_dir': ''.join(input_param['xlsx_path']),
        'test_dir': ''.join(input_param['test_dirname']),
        'wdb_name': ''.join(input_param['wandb_name']),
        'wdb_save': ''.join(input_param['wandb_save']),
        'ml_csv': ''.join(input_param['model_param_csv']),
        'dl_work':  int(''.join(input_param['dataload_workers'])),
        'a_steps': int(''.join(input_param['accumulation_steps'])),
        'num_e': int(''.join(input_param['num_epochs'])),
        'num_t': int(''.join(input_param['num_trials'])),
        'es_count': int(''.join(input_param['es_counter'])),
        'es_limit': int(''.join(input_param['es_limit'])),
        'tl_lr': list(map(float, ''.join(input_param['tl_loss_rate']).split(';'))),
        'tl_bn': ''.join(input_param['tl_batch_norm']).split(';'),
        'tl_dr': list(map(float, ''.join(input_param['tl_dropout_rate']).split(';'))),
        'tl_bs': list(map(int, ''.join(input_param['tl_batch_size']).split())),
        'tl_wd_min': float(''.join(input_param['tl_weight_decay_min'])),
        'tl_wd_max': float(''.join(input_param['tl_weight_decay_max'])),
        'tl_ga_min': float(''.join(input_param['tl_gamma_min'])),
        'tl_ga_max': float(''.join(input_param['tl_gamma_max'])),
        'tl_ga_tp': float(''.join(input_param['tl_gamma_step']))
    }
    return input_variables


def load_img_label(dataset):
    images_len = []
    labels_len = []
    ids_len = []
    for i, (image, label, study_id) in enumerate(dataset):
        img_tensor = image
        label_tensor = label
        study_id_tensor = study_id
        images_len.append(image)
        labels_len.append(label)
        ids_len.append(study_id)
    return img_tensor, label_tensor, study_id_tensor


def get_class_counts(dataset):
    class_counts = defaultdict(int)
    for idx, (img, label, _, _) in enumerate(dataset):
        try:
            class_counts[label] += 1
            if idx == len(dataset) - 1:
                print(f"Last index processed: {idx}")
        except Exception as e:
            print(f"Error processing image at index {idx}: {e}")
    return class_counts


def calculate_metrics(y_true, y_pred, y_proba, cm_label, all_id):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    metrics = {
        'acc': accuracy_score(y_true, y_pred),
        'bal_acc': balanced_accuracy_score(y_true, y_pred),
        'f1': 0.0,
        'precision': 0.0,
        'recall': 0.0
    }
    for metric, func in [('f1', f1_score), ('precision', precision_score), ('recall', recall_score)]:
        try:
            metrics[metric] = func(y_true, y_pred)
        except:
            pass
    if isinstance(y_proba, torch.Tensor):
        y_proba = y_proba.cpu().numpy()

    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    metrics['roc_plot'] = plt
    metrics['y_true'] = y_true
    metrics['y_pred'] = y_pred
    metrics['y_proba'] = y_proba
    metrics['SID'] = all_id
    metrics['fpr'] = fpr
    metrics['tpr'] = tpr
    metrics['roc_thresholds'] = thresholds
    metrics['roc_auc'] = auc(fpr, tpr)
    
    try:
        metrics['cm'] = confusion_matrix(y_true, y_pred)
    except Exception as e:
        print(f"Error computing confusion matrix: {e}")
        metrics['cm'] = None
    return metrics


def log_metrics(metrics, split, prefix, loss):
    wandb.log({
        f"{prefix}_{split}_loss": loss,
        f"{prefix}_{split}_acc": metrics['acc'],
        f"{prefix}_{split}_f1": metrics['f1'],
        f"{prefix}_{split}_balacc": metrics['bal_acc'],
        f"{prefix}_{split}_recall": metrics['recall'],
        f"{prefix}_{split}_precision": metrics['precision'],
        f"{prefix}_{split}_cnfmatrix": metrics['cm'],
        f"{prefix}_{split}_auc": metrics['roc_auc'],
    })


def get_model_paths(folder_path):
    model_names = {}
    paths = []
    names = []
    for mdl in os.listdir(folder_path):
        if mdl.endswith(".csv"):
            continue
        mdl_path = os.path.join(folder_path,mdl)
        model_name =os.path.basename(mdl_path) 
        name_split = model_name.split('_')
        model_name = name_split[4]
        model_func = name_split[3]
        epoch = int(name_split[6])
        # if model_func == 'bal':
        #     model_func = 'bal_acc'
        model_name = model_name + f'_{model_func}'
        if model_name not in model_names:
               model_names[model_name] = (epoch, mdl_path)
        else:
            if epoch > model_names[model_name][0]:
                model_names[model_name] = (epoch, mdl_path)
    for name, (epoch, path) in model_names.items():
        names.append(name+f'_epoch_{epoch}')
        paths.append(path)
    return paths, names
        
    

def create_model(model_path, model_name):
    model = densenet121(weights='DenseNet121_Weights.DEFAULT')
    for param in model.parameters():
        param.requires_grad = False
    
    for _, _, files in os.walk(input_modeldata):
        for f in files:
            if f.endswith(".csv"):
                csv_path = f"{input_modeldata}\{f}"
                input_param = {}
                with open(csv_path, 'r') as param_file:
                    details = csv.DictReader(param_file)
                    for d in details:
                        input_param.setdefault(d['eager-grass-1'], []).append(d['TRUE'])
                    param_file.close()
                for i in input_param:
                    if i == "crimson-brook-24":
                        created_model_name = i
                        model_batch_norm = input_param[i][0]
    model_path = rf'{model_path}'
    splits = model_path.split("_")
    if splits[5] == "loss":
        model_metric = "loss"
        model_epoch = splits[4]
    else:
        model_metric = "bal_acc"
        model_epoch = splits[4]
    state_dict = torch.load(model_path, map_location=device)
    num_ftrs = model.classifier.in_features
    model.ch = CustomHead(num_ftrs, 1, model_batch_norm)
    model.classifier = model.ch
    model.load_state_dict(state_dict)
    model.eval()
    model = model.to(device)
    return model, model_batch_norm, model_metric, created_model_name, model_epoch, "DenseNet121"


def test(model, test_data_loader, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    y_proba = []
    study_summary = {}
    with torch.no_grad():
        for images, labels, study_ids, img_file in test_data_loader:
            images, labels = images.to(device), labels.to(device)
            with autocast():
                outputs = model(images)
                loss = F.binary_cross_entropy_with_logits(outputs.squeeze(), labels.float())
            total_loss += loss.item() * images.size(0)
            predictions = (torch.sigmoid(outputs.squeeze()) > 0.5).int()
            # print((torch.sigmoid(outputs.squeeze()) > 0.5).int())
            # print()
            # print()
            # print((torch.sigmoid(outputs.squeeze())))
            # print()
            
        # import sys
        # sys.exit(0)
            
            all_predictions.extend(predictions.tolist())
            all_labels.extend(labels.tolist())
            probas = torch.sigmoid(outputs.squeeze().detach().cpu().float())
            y_proba.extend(probas.numpy())
            for prediction, probability, label, study_id, img_file in zip(all_predictions, probas, labels.tolist(), study_ids, img_file):
                if study_id not in study_summary:
                    study_summary[study_id] = {'predictions': [], 'probs': [], 'labels': [], 'image': []}
                study_summary[study_id]['predictions'].append(prediction)
                study_summary[study_id]['probs'].append(probability)
                study_summary[study_id]['labels'].append(label)
                study_summary[study_id]['image'].append(img_file)
        
        print("Probabilities patches:")
        print("number of:\t", len(y_proba))
        print("min:\t\t", min(y_proba))
        print("max:\t\t", max(y_proba),"\n")
        
        print("prediction patches:")
        print("number of:\t", len(all_predictions))
        print("min:\t\t", min(all_predictions))
        print("max:\t\t", max(all_predictions), "\n")
        
        print("Single study summary:")
        print(next(iter(study_summary.items())), "\n")
        
        
        print("study summary", "\n")
        print(study_summary, "\n")
        print()
        
        with open(r'\\smb01.isi01-rx.erasmusmc.nl\store_isilon\EUCRG\Shared folders\Students\2023\Maarten\Codes\MDCJ\heatmap_test\above50.csv', 'w+', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['study_id', 'predictions', 'probs', 'labels', 'image'])
            for key, values in study_summary.items():
                writer.writerow([
                    key,
                    ', '.join(map(str, values['predictions'])),
                    ', '.join(map(str, values['probs'])),
                    ', '.join(map(str, values['labels'])),
                    ', '.join(map(str, values['image'])),
                    ])
        
        patient_predictions = []
        patient_probabilities = []
        patient_labels = []
        for study_id in study_summary:
            patient_predictions.append(stats.mode(study_summary[study_id]['predictions'], keepdims=True)[0][0])
            patient_probabilities.append(np.mean(study_summary[study_id]['probs']))
            patient_labels.append(stats.mode(study_summary[study_id]['labels'], keepdims=True)[0][0])
        
        print("Probabilities patient:")
        print("number of:\t", len(patient_probabilities))
        print("min:\t\t", min(patient_probabilities))
        print("max:\t\t", max(patient_probabilities),"\n")
        
        print("prediction patient:")
        print("number of:\t", len(patient_predictions))
        print("min:\t\t", min(patient_predictions))
        print("max:\t\t", max(patient_predictions), "\n")
        
        print("label patient:")
        all_id = []
        for study_id in study_summary:
            all_id.append(study_id)
        print(all_id)
        print("labels:\t", patient_labels)
        print("number of:\t", len(patient_labels))
        print("min:\t\t", min(patient_labels))
        print("max:\t\t", max(patient_labels), "\n")
        
        img_metrics = calculate_metrics(all_labels, all_predictions, y_proba, "img", all_id)
        
        # print(img_metrics)
        
        log_metrics(img_metrics, 'test', 'img', loss)
        plt.plot(img_metrics['fpr'],img_metrics['tpr'],label='img')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        wandb.log({ "image_test_img_roc": plt})
        plt.close()
        pt_metrics = calculate_metrics(patient_labels, patient_predictions, patient_probabilities, "pt", all_id)
        
        print(pt_metrics, "\n")
        
        log_metrics(pt_metrics, 'test', 'ptnt', loss)
        plt.plot(pt_metrics['fpr'],pt_metrics['tpr'],label='ptnt')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        wandb.log({ "ptnt_test_pt_roc": plt})
        plt.close()
        return total_loss / len(test_data_loader.dataset), img_metrics, pt_metrics


def objective(trial):
    run = wandb.init(project=parameters['wdb_name'], config={})
    wandb.save(parameters['wdb_save'])
    config = run.config
    trial_lr = trial.suggest_categorical('lr', parameters['tl_lr'])
    model_path =  parameters['mdl_path']
    model_name =  parameters['mdl_name']
    print(f"\nStart testing: {model_name}\n")
    model, model_batch_norm, model_metric, created_model_name, model_epoch, model_name = create_model(model_path, model_name)
    trial_batch_norm = trial.suggest_categorical('batch_norm', model_batch_norm)
    run.config.update({
        "lr": trial_lr,
        "batch_size": trial.suggest_categorical('batch_size', parameters['tl_bs']),
        "num_epochs": parameters['num_e'],
        "weight_decay": trial.suggest_float('weight_decay', parameters['tl_wd_min'], parameters['tl_wd_max'], log=True),
        "step_size": parameters['a_steps'],
        "gamma": trial.suggest_float('gamma', parameters['tl_ga_min'], parameters['tl_ga_max'],
                                      step=parameters['tl_ga_tp']),
        "batch_norm": trial_batch_norm,
    })
    test_data_loader = DataLoader(test_dataset,
                                 batch_size=config.batch_size,
                                 shuffle=False,
                                 num_workers=parameters['dl_work'],
                                 pin_memory=True)
    config["model_name"] = model_name
    run.config.update(config)
    model.to(device)
    best_test_loss = float('inf')
    for epoch in range(parameters['num_e']):
        test_loss, img_test_metrics, pt_test_metrics = test(model, test_data_loader, device)
        bal_acc = img_test_metrics['bal_acc']
        bal_acc_pt = pt_test_metrics['bal_acc']
        trial.report(test_loss, epoch)
        print(f"Image test Metrics:\t\t"
              f"Acc: {img_test_metrics['acc']:.4f}, "
              f"F1: {img_test_metrics['f1']:.4f}, "
              f"AUC: {img_test_metrics['roc_auc']:.4f}, "
              f"Bal Acc: {bal_acc:.4f}, "
              f"Prec: {img_test_metrics['precision']:.4f}, "
              f"CM: {img_test_metrics['cm'][0]} {img_test_metrics['cm'][1]} "
              )
        
        print(f"Patient test Metrics:\t\t"
              f"Acc: {pt_test_metrics['acc']:.4f}, "
              f"F1: {pt_test_metrics['f1']:.4f}, "
              f"AUC: {pt_test_metrics['roc_auc']:.4f}, "
              f"Bal Acc: {bal_acc_pt:.4f}, "
              f"Prec: {pt_test_metrics['precision']:.4f}, "
              f"CM: {pt_test_metrics['cm'][0]} {pt_test_metrics['cm'][1]} "
              )
        
        global summary_csv
        summary_csv.loc[len(summary_csv)] = ['Patch', model_name, mdl_name, img_test_metrics['acc'], img_test_metrics['f1'], img_test_metrics['roc_auc'],
                                             bal_acc, img_test_metrics['precision'],img_test_metrics['recall'], img_test_metrics['cm'][0], img_test_metrics['cm'][1]]
        summary_csv.loc[len(summary_csv)] = ['Patient', model_name, mdl_name, pt_test_metrics['acc'], pt_test_metrics['f1'], pt_test_metrics['roc_auc'],
                                             bal_acc, pt_test_metrics['precision'],pt_test_metrics['recall'], pt_test_metrics['cm'][0], pt_test_metrics['cm'][1]]
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        print(f"Finished testing {model_name}\n")
    run.finish()
    return best_test_loss

    
def main(model_name):
    mp.set_start_method('spawn', force=True)
    wandb.init(project=parameters['wdb_name'], name=model_name)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=parameters['num_t'])


if __name__ == '__main__':
    summary_csv = pd.DataFrame(columns =['Target', 'Model', 'wandb name', 'Acc', 'F1', 'AUC', 'Bal Acc', 'Precision', 'Recall', 'CM-1', 'CM-2'])
    model_paths, model_names = get_model_paths(models_path)
    for mdl_path, mdl_name in zip(model_paths, model_names):
        mp.set_start_method('spawn', force=True)
        parameters = load_parameters(param_path)
        root_dir = parameters['root_dir']
        xlsx_dir = parameters['xlsx_dir']
        parameters['mdl_path'] = mdl_path
        parameters['mdl_name'] = mdl_name
        wandb_model_name_split = mdl_name.split('_')[2:-2]
        wandb_model_name = '_'.join(wandb_model_name_split)
        transform = transforms.Compose([transforms.ToTensor()])
        test_dataset = CustomImageDataset(os.path.join(parameters['root_dir'],
                                                       parameters['test_dir']),
                                          parameters['xlsx_dir'],
                                          transform=transform)
        print("\nTest index processing:")
        test_class_counts = get_class_counts(test_dataset)
        print("\nTest set class counts:")
        for label, count in test_class_counts.items():
            print(f"Class {label}: {count} images")
        print("\nChecking for GPU availability...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)
        if not os.path.exists('Best models'):
            os.makedirs('Best models')
        print("\nStarting main\n")
        main(wandb_model_name)
        print("Main complete")
        summary_csv.to_csv(os.path.join(models_path, 'testset_performance_updated.csv'))
        print("Summary CSV complete")
