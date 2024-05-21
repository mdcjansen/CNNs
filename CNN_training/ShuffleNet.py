#!/usr/bin/env python3

import csv
import matplotlib.pyplot as plt
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
from torchvision.models import shufflenet_v2_x1_0

# Credentials
__author__ = "M.D.C. Jansen"
__version__ = "1.7"
__date__ = "14/05/2023"

# Parameter file path
param_path = r"D:/path/to/hyperparameter.csv"


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
        return image, label, study_id


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


def generate_filename(prefix, run_name, epoch, cfbatchnorm):
    epochstr= str(epoch + 1)
    return os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        os.path.join("Best models", f"{prefix}_{run_name}_epoch_{epochstr}_batch_norm_{cfbatchnorm}.pt"))


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
        'train_dir': ''.join(input_param['train_dirname']),
        'val_dir': ''.join(input_param['val_dirname']),
        'wdb_name': ''.join(input_param['wandb_name']),
        'wdb_save': ''.join(input_param['wandb_save']),
        'ml_csv': ''.join(input_param['model_param_csv']),
        'dl_work':  int(''.join(input_param['dataload_workers'])),
        'a_steps': int(''.join(input_param['accumulation_steps'])),
        'num_e': int(''.join(input_param['num_epochs'])),
        'num_t': int(''.join(input_param['num_trials'])),
        'es_count': int(''.join(input_param['es_counter'])),
        'es_limit': int(''.join(input_param['es_limit'])),
        'tl_lr': list(map(float, ''.join(input_param['tl_learn_rate']).split(';'))),
        'tl_bn': ''.join(input_param['tl_batch_norm']).split(';'),
        'tl_bs_min': int( ''.join(input_param['tl_batch_size_min'])),
        'tl_bs_max': int( ''.join(input_param['tl_batch_size_max'])),
        'tl_bs_tp': int(''.join(input_param['tl_batch_size_step'])),
        'tl_wd_min': float(''.join(input_param['tl_weight_decay_min'])),
        'tl_wd_max': float(''.join(input_param['tl_weight_decay_max'])),
        'tl_ga_min': float(''.join(input_param['tl_gamma_min'])),
        'tl_ga_max': float(''.join(input_param['tl_gamma_max'])),
        'tl_ga_tp': float(''.join(input_param['tl_gamma_step']))
    }
    return input_variables


def custom_transform(image):
    rd_colour_change = random.random() < 0.3
    if rd_colour_change:
        brightness = random.uniform(0.0, 0.25)
        contrast = random.uniform(0.0, 0.2)
        saturation = random.uniform(0.0, 0.4)
        hue = random.uniform(0.0, 0.5)
        image = transforms.functional.adjust_brightness(image, brightness)
        image = transforms.functional.adjust_contrast(image, contrast)
        image = transforms.functional.adjust_saturation(image, saturation)
        image = transforms.functional.adjust_hue(image, hue)
    return transforms.functional.to_tensor(image)


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
    for idx, (img, label, _) in enumerate(dataset):
        try:
            class_counts[label] += 1
            if idx == len(dataset) - 1:
                print(f"Last index processed: {idx}")
        except Exception as e:
            print(f"Error processing image at index {idx}: {e}")
    return class_counts


def calculate_metrics(y_true, y_pred, y_proba):
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
        f"{prefix}_{split}_auc": metrics['roc_auc']
    })


def create_model(batch_norm):
    model = shufflenet_v2_x1_0(weights="ShuffleNet_V2_X1_0_Weights.DEFAULT")
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.ch = CustomHead(num_ftrs, 1, batch_norm)
    model.fc = model.ch
    for param in model.parameters():
        param.requires_grad = True
    model = model.to(device)
    return model, "ShuffleNet V2 x1.0"


def train(model, train_data_loader, optimizer, scheduler, device, scaler):
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []
    y_logits = []
    optimizer.zero_grad()
    for i, (images, labels, study_ids) in enumerate(train_data_loader):
        images, labels = images.to(device), labels.to(device)
        with autocast():
            outputs = model(images)
            loss = F.binary_cross_entropy_with_logits(outputs.squeeze(), labels.float())
            scaler.scale(loss).backward()
            loss_item = loss.item()
            output_detach = outputs.detach()
            del loss, outputs
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        all_predictions.extend((torch.sigmoid(output_detach.squeeze()) > 0.5).tolist())
        all_labels.extend(labels.tolist())
        y_logits.extend(output_detach.squeeze().cpu().numpy())
        total_loss += loss_item * images.size(0)
        torch.cuda.empty_cache()
    scheduler.step()
    metrics = calculate_metrics(all_labels, all_predictions, y_logits)
    _, _, train_patient_predictions, train_patient_labels, _, train_patient_probs = predict(model, train_data_loader)
    train_patient_metrics = calculate_metrics(train_patient_labels, train_patient_predictions, train_patient_probs)
    del all_predictions, all_labels, y_logits
    torch.cuda.empty_cache()
    return total_loss / len(train_data_loader.dataset), metrics, train_patient_metrics


def validate(model, val_data_loader, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    y_proba = []
    with torch.no_grad():
        for images, labels, study_ids in val_data_loader:
            images, labels = images.to(device), labels.to(device)
            with autocast():
                outputs = model(images)
                loss = F.binary_cross_entropy_with_logits(outputs.squeeze(), labels.float())
                loss_item = loss.item()
                output_detach = outputs.detach()
                del loss, outputs
            total_loss += loss_item * images.size(0)
            predictions = (torch.sigmoid(output_detach.squeeze()) > 0.5)
            all_predictions.extend(predictions.tolist())
            all_labels.extend(labels.tolist())
            probas = torch.sigmoid(output_detach.squeeze().cpu().float())
            y_proba.extend(probas.numpy())
            torch.cuda.empty_cache()
    metrics = calculate_metrics(all_labels, all_predictions, y_proba)
    del all_predictions, all_labels, y_proba
    torch.cuda.empty_cache()
    return total_loss / len(val_data_loader.dataset), metrics


def predict(model, data_loader):
    model.eval()
    model = model.to(device)
    batch_predictions = []
    batch_labels = []
    study_summary = {}
    y_proba = []
    with torch.no_grad():
        for i, (inputs, labels, study_ids) in enumerate(data_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            logits = outputs
            output = logits.view(-1)
            output_detach = output.detach()
            del outputs, logits, output
            probs = torch.sigmoid(output_detach.cpu().float()).numpy()
            y_proba.extend(probs)
            batch_predictions.extend((probs > 0.25).astype(int).tolist())
            batch_labels.extend(labels.tolist())
            for prediction, probability, label, study_id in zip(batch_predictions, probs, labels.tolist(), study_ids):
                if study_id not in study_summary:
                    study_summary[study_id] = {'predictions': [], 'probs': [], 'labels': []}
                study_summary[study_id]['predictions'].append(prediction)
                study_summary[study_id]['probs'].append(probability)
                study_summary[study_id]['labels'].append(label)
            torch.cuda.empty_cache()
        patient_predictions = []
        patient_probabilities = []
        patient_labels = []
        for study_id in study_summary:
            patient_predictions.append(stats.mode(study_summary[study_id]['predictions'], keepdims=True)[0][0])
            patient_probabilities.append(np.mean(study_summary[study_id]['probs']))
            patient_labels.append(stats.mode(study_summary[study_id]['labels'], keepdims=True)[0][0])
        torch.cuda.empty_cache()
        return batch_predictions, batch_labels, patient_predictions, patient_labels, y_proba, patient_probabilities


def objective(trial):
    run = wandb.init(project=parameters['wdb_name'], config={})
    wandb.save(parameters['wdb_save'])
    config = run.config
    print(f"\nStarting trial {trial.number}")
    trial_lr = trial.suggest_categorical('lr', parameters['tl_lr'])
    trial_batch_norm = trial.suggest_categorical('batch_norm', parameters['tl_bn'])
    run.config.update({
        "lr": trial_lr,
        "batch_size": trial.suggest_int('batch_size', parameters['tl_bs_min'], parameters["tl_bs_max"],
                                          step=parameters["tl_bs_tp"]),
        "num_epochs": parameters['num_e'],
        "weight_decay": trial.suggest_float('weight_decay', parameters['tl_wd_min'], parameters['tl_wd_max'], log=True),
        "step_size": parameters['a_steps'],
        "gamma": trial.suggest_float('gamma', parameters['tl_ga_min'], parameters['tl_ga_max'],
                                     step=parameters['tl_ga_tp']),
        "batch_norm": trial_batch_norm,
    }, allow_val_change=True)
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=config.batch_size,
                                   shuffle=True,
                                   num_workers=parameters['dl_work'],
                                   pin_memory=True)
    val_data_loader = DataLoader(val_dataset,
                                 batch_size=config.batch_size,
                                 shuffle=False,
                                 num_workers=parameters['dl_work'],
                                 pin_memory=True)
    model, model_name = create_model(config["batch_norm"])
    config["model_name"] = model_name
    run.config.update(config, allow_val_change=True)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
    scaler = GradScaler()
    best_val_loss = float('inf')
    best_patient_auc = float('-inf')
    best_patient_bal_acc = float('-inf')
    early_stop_counter = parameters['es_count']
    early_stop_limit = parameters['es_limit']
    for epoch in range(config["num_epochs"]):
        print(f"\nEpoch {epoch + 1} - Started")
        training_loss, training_metrics, train_patient_metrics = train(model, train_data_loader, optimizer, scheduler,
                                                                       device, scaler)
        log_metrics(training_metrics, 'train', 'img', training_loss)
        log_metrics(train_patient_metrics, 'train', 'ptnt', None)
        plt.plot(training_metrics['fpr'],training_metrics['tpr'],label='Training_patch')
        plt.title('ROC_AUC Training patch level')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        wandb.log({"training_img_roc": plt})
        plt.close()
        plt.plot(train_patient_metrics['fpr'],train_patient_metrics['tpr'],label='Training_patient')
        plt.title('ROC_AUC Training patient level')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        wandb.log({"training_ptnt_roc": plt})
        plt.close()
        print(f"Epoch {epoch + 1} - Training Metrics:\t\t"
              f"AUC: {training_metrics['roc_auc']:.4f}, "
              f"Bal Acc: {training_metrics['bal_acc']:.4f}, "
              f"Loss: {training_loss:.4f} "
              )
        validation_loss, validation_metrics = validate(model, val_data_loader, device)
        log_metrics(validation_metrics, 'val', 'img', validation_loss)
        plt.plot(validation_metrics['fpr'],validation_metrics['tpr'],label='Validation_patch')
        plt.title('ROC_AUC Validation patch level')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        wandb.log({"validation_img_roc": plt})
        plt.close()
        print(f"Epoch {epoch + 1} - Validation Metrics:\t\t"
              f"AUC: {validation_metrics['roc_auc']:.4f}, "
              f"Bal Acc: {validation_metrics['bal_acc']:.4f}, "
              f"Loss: {validation_loss:.4f} "
              )
        batch_predictions, batch_labels, patient_predictions, patient_labels, y_proba, patient_probs = predict(model,
                                                                                                               val_data_loader)
        patient_metrics = calculate_metrics(patient_labels, patient_predictions, patient_probs)
        log_metrics(patient_metrics, 'val', 'ptnt', None)
        plt.plot(patient_metrics['fpr'],patient_metrics['tpr'],label='Validation_patient')
        plt.title('ROC_AUC Validation patient level')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        wandb.log({"validation_ptnt_roc": plt})
        plt.close()
        print(f"Epoch {epoch + 1} - Patient-Level Metrics:\t"
              f"AUC: {patient_metrics['roc_auc']:.4f}, "
              f"Bal Acc: {patient_metrics['bal_acc']:.4f}"
              )
        is_best_loss = validation_loss < best_val_loss
        is_best_bal_acc = patient_metrics['bal_acc'] > best_patient_bal_acc
        is_best_patient_auc = patient_metrics['roc_auc'] > best_patient_auc
        if is_best_loss:
            best_val_loss = validation_loss
            torch.save(model.state_dict(), generate_filename('best_model_val_loss', run.name, epoch, config["batch_norm"]))
            with open(parameters['ml_csv'], 'a+') as model_csv:
                ml_params = [f"{key}={value}" for key, value in config.items() if key not in ['project']]
                model_csv.write(f"{run.name},loss,epoch={epoch + 1},{','.join(ml_params)}\n")
            model_csv.close()
        if is_best_bal_acc:
            best_patient_bal_acc = patient_metrics['bal_acc']
            torch.save(model.state_dict(), generate_filename('best_model_ptnt_balacc', run.name, epoch, config["batch_norm"]))
            with open(parameters['ml_csv'], 'a+') as model_csv:
                ml_params = [f"{key}={value}" for key, value in config.items() if key not in ['project']]
                model_csv.write(f"{run.name},loss,epoch={epoch + 1},{','.join(ml_params)}\n")
            model_csv.close()
        if is_best_patient_auc:
            best_patient_auc = patient_metrics['roc_auc']
            torch.save(model.state_dict(), generate_filename('best_model_ptnt_auc', run.name, epoch, config["batch_norm"]))
            with open(parameters['ml_csv'], 'a+') as model_csv:
                ml_params = [f"{key}={value}" for key, value in config.items() if key not in ['project']]
                model_csv.write(f"{run.name},loss,epoch={epoch + 1},{','.join(ml_params)}\n")
            model_csv.close()
        if not (is_best_loss or is_best_bal_acc or is_best_patient_auc):
            early_stop_counter += 1
            if early_stop_counter >= early_stop_limit:
                print('Early stopping triggered')
                print("\n\n\n")
                break
        else:
            early_stop_counter = 0
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        print(f"Epoch {epoch + 1} - Finished\n")
    run.finish()
    return best_val_loss


def main():
    mp.set_start_method('spawn', force=True)
    wandb.init(project=parameters['wdb_name'])
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=parameters['num_t'])


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    parameters = load_parameters(param_path)
    root_dir = parameters['root_dir']
    xlsx_dir = parameters['xlsx_dir']
    transform = transforms.Compose([custom_transform])
    # transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = CustomImageDataset(os.path.join(parameters['root_dir'],
                                                    parameters['train_dir']),
                                       parameters['xlsx_dir'],
                                       transform=transform)
    val_dataset = CustomImageDataset(os.path.join(parameters['root_dir'],
                                                  parameters['val_dir']),
                                     parameters['xlsx_dir'],
                                     transform=transform)
    print("\nTraining index processing:")
    train_class_counts = get_class_counts(train_dataset)
    print("\nValidation index processing:")
    val_class_counts = get_class_counts(val_dataset)
    print("\nTraining set class counts:")
    for label, count in train_class_counts.items():
        print(f"Class {label}: {count} images")
    print("\nValidation set class counts:")
    for label, count in val_class_counts.items():
        print(f"Class {label}: {count} images")
    print("\nChecking for GPU availability...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if not os.path.exists('Best models'):
        os.makedirs('Best models')
    print("\nStarting main\n")
    main()
    print("Main complete")
