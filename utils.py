import numpy as np 
import pandas as pd 

import os
import gc
import time
import random
from torch.nn.modules import activation
from tqdm import tqdm

import sklearn
from sklearn.model_selection import train_test_split

import torch
import sys
sys.path.append("/home/toefl/K/SETI/pytorch-image-models-master")
import timm
import torchvision
import torch.nn as nn
from torch.cuda import amp

import matplotlib.pyplot as plt
import wandb

from custom.model import Net
from custom.loss import ArcFaceLoss, ArcFaceLossAdaptiveMargin
from custom.optimizer import Ranger
from data import get_loaders, get_train_file_path
from custom.scheduler import GradualWarmupSchedulerV2
from custom.augmentations import *


def gap(y_true, y_pred, num_classes, ignore_non_landmarks=True):
    indexes = np.argsort(y_pred[1])[::-1]
    queries_with_target = (y_true < num_classes).sum()
    correct_predictions = 0
    total_score = 0.
    i = 1
    for k in indexes:
        if ignore_non_landmarks and y_true[k] == num_classes:
            continue
        if y_pred[0][k] == num_classes:
            continue
        relevance_of_prediction_i = 0
        if y_true[k] == y_pred[0][k]:
            correct_predictions += 1
            relevance_of_prediction_i = 1
        precision_at_rank_i = correct_predictions / i
        total_score += precision_at_rank_i * relevance_of_prediction_i
        i += 1
    return 1 / queries_with_target * total_score


def set_seed(seed: int):
    '''Set a random seed for complete reproducibility.'''

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def print_size_of_model(model):
    """Gets model's size."""
    torch.save(model.state_dict(), "temp.p")
    print("Model's Size (MB):", os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')


def calibrate_model(config, model, num_batches):
    """Used for model calibration after applying optimization methods."""
    with torch.no_grad():
        for _ in tqdm(range(num_batches)):
            inputs = torch.randn(config.inference.batch_size, 3,
                                  config.data.final_size, config.data.final_size).to(config.inference.device)
            _ = model(inputs)


def optimize_model(config):
    """Apply optimization for model."""

    model = get_model(config, type='classic')

    try:
        model.load_state_dict(torch.load(config.optimization_params.path_to_load)["model"])
    except KeyError:
        model.load_state_dict(torch.load(config.optimization_params.path_to_load))

    model.to(config.inference.device)
    model.eval()

    print_size_of_model(model)
    print("Calibrating...")
    
    if config.optimization_params.fuse_model:
        model.fuse_model()

    if config.optimization_params.static_quantization and config.inference.device == "cpu":
        model.qconfig = torch.quantization.default_qconfig
        torch.quantization.prepare(model, inplace=True)

        calibrate_model(config, model, num_batches=100)

        torch.quantization.convert(model, inplace=True)
    
    torch.save(model.state_dict(), config.optimization_params.path_to_save)
    print("Calibration is finished!")
    print_size_of_model(model)


def get_model(config, type):
    '''Get PyTorch model.'''
    if type == "student":

        print('Model:', config.student_model.name)

        if config.student_model.name.startswith('/timm/'): 
            model = timm.create_model(config.student_model.name[6:], pretrained=config.student_model.pretrained)
        elif config.student_model.name.startswith('/torch/'):
            model = getattr(torchvision.models, config.student_model.name[7:])(pretrained=config.student_model.pretrained)
        elif config.student_model.name.startswith('/custom/'):
            model = globals()[config.student_model.name[8:]](**config.student_model.params)
        else:
            raise RuntimeError('Unknown model source. Use /timm/ or /torch/.')

        last_layer = list(model._modules)[-1]
        try:
            setattr(model, last_layer, nn.Linear(in_features=getattr(model, last_layer).in_features,
                                                out_features=config.general.num_classes, bias=True))
        except AttributeError:
            setattr(model, last_layer, nn.Linear(in_features=getattr(model, last_layer)[1].in_features,
                                                out_features=config.general.num_classes, bias=True))

    elif type == "teacher":

        print('Model:', config.teacher_model.name)

        if config.teacher_model.name.startswith('/timm/'): 
            model = timm.create_model(config.teacher_model.name[6:], pretrained=config.teacher_model.pretrained)
        elif config.teacher_model.name.startswith('/torch/'):
            model = getattr(torchvision.models, config.teacher_model.name[7:])(pretrained=config.teacher_model.pretrained)
        elif config.teacher_model.name.startswith('/custom/'):
            model = globals()[config.teacher_model.name[8:]](**config.teacher_model.params)
        else:
            raise RuntimeError('Unknown model source. Use /timm/ or /torch/.')

        last_layer = list(model._modules)[-1]
        try:
            setattr(model, last_layer, nn.Linear(in_features=getattr(model, last_layer).in_features,
                                                out_features=config.general.num_classes, bias=True))
        except AttributeError:
            setattr(model, last_layer, nn.Linear(in_features=getattr(model, last_layer)[1].in_features,
                                                out_features=config.general.num_classes, bias=True))

    elif type == "classic":
        print('Model:', config.model.name)

        if config.model.name.startswith('/timm/'): 
            model = timm.create_model(config.model.name[6:], pretrained=config.model.pretrained)
        elif config.model.name.startswith('/torch/'):
            model = getattr(torchvision.models, config.model.name[7:])(pretrained=config.model.pretrained)
        elif config.model.name.startswith('/custom/'):
            model = globals()[config.model.name[8:]](**config.model.params)
        else:
            raise RuntimeError('Unknown model source. Use /timm/ or /torch/.')

        last_layer = list(model._modules)[-1]
        try:
            setattr(model, last_layer, nn.Linear(in_features=getattr(model, last_layer).in_features,
                                                out_features=config.general.num_classes, bias=True))
        except AttributeError:
            setattr(model, last_layer, nn.Linear(in_features=getattr(model, last_layer)[1].in_features,
                                                out_features=config.general.num_classes, bias=True))

    else:
        raise ValueError()

    model = model.to(config.training.device)

    return model


def get_optimizer(config, model):
    '''Get PyTorch optimizer'''

    if config.optimizer.name.startswith('/custom/'):
        optimizer = globals()[config.optimizer.name[8:]](model.parameters(), **config.optimizer.params)
    else:
        optimizer = getattr(torch.optim, config.optimizer.name)(model.parameters(), **config.optimizer.params)
    
    if config.resume_from_checkpoint.optimizer_state is not None:
        optimizer.load_state_dict(config.resume_from_checkpoint.optimizer_stat)

    return optimizer


def get_scheduler(config, optimizer):
    '''Get PyTorch scheduler'''

    if config.scheduler.name.startswith('/custom/'):
        scheduler = globals()[config.scheduler.name[8:]](optimizer, **config.scheduler.params)
    else:
        scheduler = getattr(torch.optim.lr_scheduler, config.scheduler.name)(optimizer, **config.scheduler.params)
    
    if config.training.warmup_scheduler:
        final_scheduler = GradualWarmupSchedulerV2(optimizer, multiplier=config.training.warmup_multiplier,
                                                 total_epoch=config.training.warmup_epochs, after_scheduler=scheduler)

        return final_scheduler
    else:
        return scheduler

    
def get_loss(config, type):
    '''Get PyTorch loss function.'''

    if type == "classic":
        if config.loss.classic_loss_name.startswith('/custom/'):
            loss = globals()[config.loss.classic_loss_name[8:]](**config.loss.classic_loss_params)
        else:
            loss = getattr(nn, config.loss.classic_loss_name)(**config.loss.classic_loss_params)

    elif type == "validation":
        if config.loss.val_loss_name.startswith('/custom/'):
            loss = globals()[config.loss.val_loss_name[8:]](**config.loss.val_loss_params)
        else:
            loss = getattr(nn, config.loss.val_loss_name)(**config.loss.val_loss_params)

    elif type == "hard":
        if config.loss.hard_loss_name.startswith('/custom/'):
            loss = globals()[config.loss.hard_loss_name[8:]](**config.loss.hard_loss_params)
        else:
            loss = getattr(nn, config.loss.hard_loss_name)(**config.loss.hard_loss_params)

    elif type == "soft":
        if config.loss.soft_loss_name.startswith('/custom/'):
            loss = globals()[config.loss.soft_loss_name[8:]](**config.loss.soft_loss_params)
        else:
            loss = getattr(nn, config.loss.soft_loss_name)(**config.loss.soft_loss_params)

    else:
        raise ValueError("Invalid loss type.")

    return loss


def get_metric(config, y_true, y_pred, conf):
    '''Calculate metric.'''
    
    print("Counting")

    if not config.training.train_binary_classifier:
        predictions = [y_pred, conf]
    else:
        predictions = y_pred

    if config.metric.name.startswith('/custom/'):
        score = globals()[config.metric.name[8:]](y_true, predictions, **config.metric.params)
    else:
        score = getattr(sklearn.metrics, config.metric.name)(y_true, predictions, **config.metric.params)
    
    return score


def train(config, model, train_loader, optimizer, scheduler, loss_function, epoch, image_size, scaler):
    '''Train loop.'''

    print('Training')

    model.train()

    if config.model.freeze_batchnorms:
        for name, child in (model.named_children()):
            if name.find('BatchNorm') != -1:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True
    
    total_loss = 0.0

    for step, batch in enumerate(tqdm(train_loader)):
        inputs, labels = batch
        inputs, labels = inputs.to(config.training.device), labels.squeeze(1).to(config.training.device) # remove???

        p = random.uniform(0, 1)
        flag = False
        if p < 0.25 and config.data.fmix and epoch > config.training.warmup_epochs:
            inputs, labels = fmix(inputs, labels, 3.0, 1.0, (image_size, image_size))
            flag = True
        elif 0.25 <= p < 0.5 and config.data.cutmix and epoch > config.training.warmup_epochs:
            inputs, labels = cutmix(inputs, labels, alpha=1.0)
            flag = True
        elif 0.5 <= p < 0.75 and config.data.mixup and epoch > config.training.warmup_epochs:
            inputs, labels = mixup(inputs, labels, alpha=0.2)
            flag = True

        if not config.training.gradient_accumulation:
            optimizer.zero_grad()
        
        if config.training.mixed_precision:
            with amp.autocast():

                outputs = model(inputs.float()).squeeze(1)
                if flag:
                    loss = loss_function(outputs, labels['target']) * labels['lam'] + loss_function(outputs,
                                                                                        labels['shuffled_target']) * (
                                        1.0 - labels['lam'])
                else:
                    loss = loss_function(outputs, labels)


                if config.training.gradient_accumulation:
                    loss = loss / config.training.gradient_accumulation_steps
        else:
            outputs = model(inputs.float()).squeeze(1)
            if flag:
                loss = loss_function(outputs, labels['target']) * labels['lam'] + loss_function(outputs,
                                                                                    labels['shuffled_target']) * (
                                    1.0 - labels['lam'])
            else:
                loss = loss_function(outputs, labels)

        total_loss += loss.item()

        if config.training.mixed_precision:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if config.training.gradient_clipping:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.clip_value)

        if config.training.gradient_accumulation:
            if (step + 1) % config.training.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        elif config.training.mixed_precision:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        if config.scheduler.interval == 'step':
            if config.training.warmup_scheduler:
                if epoch >= config.training.warmup_epochs:
                    scheduler.step()
            else:
                scheduler.step()
    
    scheduler.step()
    # if config.training.warmup_scheduler:
    #     if epoch < config.training.warmup_epochs:
    #         scheduler.step()
    #     elif epoch > config.training.warmup_epochs:
    #         if config.scheduler.interval == 'epoch':
    #             scheduler.step()
    # else:
    #     if config.scheduler.interval == 'epoch':
    #         scheduler.step()

    print('Learning rate:', optimizer.param_groups[0]['lr'])

    return total_loss / len(train_loader)


def validation(config, model, val_loader, loss_function):
    '''Validation loop.'''

    print('Validating')

    model.eval()

    total_loss = 0.0
    activation = torch.nn.Softmax(dim=1)
    preds, conf, targets = [], [], []

    with torch.no_grad():
        for step, batch in enumerate(tqdm(val_loader)):
            inputs, labels = batch
            inputs, labels = inputs.to(config.training.device), labels.squeeze(1).to(config.training.device)

            outputs = model(inputs)

            preds.append(activation(outputs).argmax(dim=1).tolist())
            targets.append(labels.to('cpu').numpy())
            conf.append(activation(outputs).to('cpu').numpy().max(axis=1).tolist())

            loss = loss_function(outputs, labels)
            total_loss += loss.item()

    return total_loss / len(val_loader), np.concatenate(preds), np.concatenate(targets), np.concatenate(conf)


def test(config, model, test_loader, loss_function):
    '''Testing loop.'''

    print('Testing')

    model.eval()

    total_loss = 0.0

    preds, targets = [], []

    with torch.no_grad():
        for step, batch in enumerate(tqdm(test_loader)):
            inputs, labels = batch
            inputs, labels = inputs.to(config.training.device), labels.squeeze(1).to(config.training.device)

            outputs = model(inputs)

            preds.append(outputs.sigmoid().to('cpu').numpy())
            targets.append(labels.to('cpu').numpy())

            loss = loss_function(outputs, labels)
            total_loss += loss.item()

    metric = get_metric(np.concatenate(targets), np.concatenate(preds))
    print('Test Loss:', total_loss / len(test_loader), '\nTest Metric:', metric)


def classic_training(config):
    '''Main function.'''

    # Logging
    if config.logging.log:
        wandb.init(project=config.logging.wandb_project_name)

    # Create working directory
    if not os.path.exists(config.paths.path_to_checkpoints):
        os.makedirs(config.paths.path_to_checkpoints)

    if config.data.test_size == 0.0:
        train_loader, val_loader = get_loaders(config)
    else:
        train_loader, val_loader, test_loader = get_loaders(config)

    torch.cuda.empty_cache()

    # Get objects
    if config.model.name == "Net":
        model = Net(config.model.params, pretrained=config.model.pretrained).to(config.training.device)
    else:
        model = get_model(config, type="classic")
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)


    if config.training.mixed_precision:
        scaler = amp.GradScaler()
    else:
        scaler = None

    if config.resume_from_checkpoint.resume:
        cp = torch.load(config.paths.path_to_weights)

        scaler.load_state_dict(cp["scaler"])
        model.load_state_dict(cp["model"])
        optimizer.load_state_dict(cp["optimizer"])
        scheduler = get_scheduler(config, optimizer)
        for _ in range(cp["epoch"]):
            scheduler.step()

        config.resume_from_checkpoint.epochs_since_improvement = cp["epochs_since_improvement"]
        config.resume_from_checkpoint.last_epoch = cp["epoch"]
    
        del cp

    if not config.training.train_binary_classifier:
        df = pd.read_csv(config.paths.path_to_csv)
        df, _, _, _ = train_test_split(df,
                                    df[config.data.target_columns],
                                    train_size=config.data.train_size,
                                    test_size=config.data.val_size,
                                    random_state=config.general.seed,
                                    stratify=df[config.data.target_columns])

        df = df.drop_duplicates()

        tmp = np.sqrt(1 / np.sqrt(df['landmark_id'].value_counts().sort_index().values))
        margins = (tmp - tmp.min()) / (tmp.max() - tmp.min()) * 0.45 + 0.05
        loss_function = ArcFaceLossAdaptiveMargin(margins=margins)
        del df
    else:
        loss_function = get_loss(config, type="classic")

    # Initializing metrics and logging
    train_losses, val_losses, metrics, learning_rates = [], [], [], []
    best_metric = 0

    if config.resume_from_checkpoint.epochs_since_improvement:
        epochs_since_improvement = config.resume_from_checkpoint.epochs_since_improvement
    else:
        epochs_since_improvement = 0
    
    print('Testing ' + config.general.experiment_name + ' approach')
    if config.paths.log_name:
        with open(os.path.join(config.paths.path_to_checkpoints, config.paths.log_name), 'w') as file:
            file.write('Testing ' + config.general.experiment_name + ' approach\n')
    
    # Training
    # Store transforms to use them after warmup stage
    transforms = config.augmentations.transforms
    image_size = config.data.start_size
    
    if config.resume_from_checkpoint.last_epoch > 0:
        current_epoch = config.resume_from_checkpoint.last_epoch
    else:
        current_epoch = 0

    for epoch in range(current_epoch, config.training.num_epochs):
        print('\nEpoch: ' + str(epoch + 1))

        if not config.training.train_binary_classifier:
            if epoch in [0, 1, 2]:
                config.data.batch_size = 32
                config.data.first_size = 224
                config.data.aftercrop_size = 224

                config.data.cutmix = False
                config.data.mixup = False
                config.data.fmix = False
                config.data.pre_transforms = [
                    {
                        'name': 'Resize',
                        'params': {
                            'height': config.data.aftercrop_size,
                            'width': config.data.aftercrop_size,
                            'p': 1.0
                        }
                    },
                ]

                config.augmentations.transforms = [
                    {
                        'name': 'HorizontalFlip',
                        'params': {
                            'p': 0.5
                        }
                    }
                ]

            elif epoch in [3, 4, 5]:
                config.data.batch_size = 8
                config.data.first_size = 512
                config.data.aftercrop_size = 448 

                config.data.cutmix = False
                config.data.mixup = False
                config.data.fmix = False
                config.data.pre_transforms = [
                    {
                        'name': 'Resize',
                        'params': {
                            'height': config.data.first_size,
                            'width': config.data.first_size,
                            'p': 1.0
                        }
                    },
                ]

                config.augmentations.transforms = [
                    {
                        'name': 'RandomCrop',
                        'params': {
                            'height': config.data.aftercrop_size,
                            'width': config.data.aftercrop_size,
                            'p': 1.0
                        }
                    },
                    {
                        'name': 'HorizontalFlip',
                        'params': {
                            'p': 0.5
                        }
                    },
                ]

            elif epoch in [6, 7, 8]:
                config.data.batch_size = 8
                config.data.first_size = 512
                config.data.aftercrop_size = 448

                config.data.cutmix = False
                config.data.mixup = False
                config.data.fmix = False
                config.data.pre_transforms = [
                    {
                        'name': 'Resize',
                        'params': {
                            'height': config.data.first_size,
                            'width': config.data.first_size,
                            'p': 1.0
                        }
                    },
                ]

                config.augmentations.transforms = [
                    {
                        'name': 'RandomCrop',
                        'params': {
                            'height': config.data.aftercrop_size,
                            'width': config.data.aftercrop_size,
                            'p': 1.0
                        }
                    },
                    {
                        'name': 'HorizontalFlip',
                        'params': {
                            'p': 0.5
                        }
                    },
                    {
                        "name": "CoarseDropout",
                        "params": {
                        "max_holes": 1,
                        "max_height": 64,
                        "max_width": 64,
                        "min_holes": 1,
                        "min_height": 64,
                        "min_width": 64,
                        "p": 0.5,
                    }
                },
            ]

            elif epoch in [9, 10, 11]:
                config.data.batch_size = 8
                config.data.first_size = 512
                config.data.aftercrop_size = 448

                config.data.cutmix = False
                config.data.mixup = False
                config.data.fmix = False
                config.data.pre_transforms = [
                    {
                        'name': 'Resize',
                        'params': {
                            'height': config.data.first_size,
                            'width': config.data.first_size,
                            'p': 1.0
                        }
                    },
                ]

                config.augmentations.transforms = [
                    {
                        'name': 'RandomCrop',
                        'params': {
                            'height': config.data.aftercrop_size,
                            'width': config.data.aftercrop_size,
                            'p': 1.0
                        }
                    },
                    {
                        'name': 'HorizontalFlip',
                        'params': {
                            'p': 0.5
                        }
                    },
                ]

            elif epoch in [12, 13, 14]:
                config.data.batch_size = 4
                config.data.first_size = 704
                config.data.aftercrop_size = 640

                config.data.cutmix = False
                config.data.mixup = False
                config.data.fmix = False
                config.data.pre_transforms = [
                    {
                        'name': 'Resize',
                        'params': {
                            'height': config.data.first_size,
                            'width': config.data.first_size,
                            'p': 1.0
                        }
                    },
                ]

                config.augmentations.transforms = [
                    {
                        'name': 'RandomCrop',
                        'params': {
                            'height': config.data.aftercrop_size,
                            'width': config.data.aftercrop_size,
                            'p': 1.0
                        }
                    },
                    {
                        'name': 'HorizontalFlip',
                        'params': {
                            'p': 0.5
                        }
                    },
                ]

            print(config.data.first_size)
            print(config.data.aftercrop_size)

        if config.data.test_size == 0.0:
            train_loader, val_loader = get_loaders(config)
        else:
            train_loader, val_loader, test_loader = get_loaders(config)

        image_size = config.data.aftercrop_size

        print('Image size: ' + str(image_size))

        start_time = time.time()

        train_loss = train(config, model, train_loader, optimizer, scheduler, loss_function, epoch, image_size, scaler)
        val_loss, predictions, targets, conf = validation(config, model, val_loader, loss_function)
        current_metric = get_metric(config, targets, predictions, conf)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        metrics.append(current_metric)
        learning_rates.append(optimizer.param_groups[0]['lr'])

        if current_metric > best_metric:
            print('New Record!')

            epochs_since_improvement = 0
            best_metric = current_metric
            
            save_model(config, model, epoch, train_loss, val_loss, current_metric, optimizer, epochs_since_improvement, 
                       'best.pt', scheduler, image_size, scaler)
        else:
            epochs_since_improvement += 1

        if epoch % config.training.save_step == 0:
            save_model(config, model, epoch, train_loss, val_loss, current_metric, optimizer, epochs_since_improvement,
                       f'{epoch + 1}_epoch.pt', scheduler, image_size, scaler)

        t = int(time.time() - start_time)
        print_report(t, train_loss, val_loss, current_metric, best_metric)

        if config.paths.log_name:
            save_log(os.path.join(config.paths.path_to_checkpoints, config.paths.log_name), epoch + 1,
                    train_loss ,val_loss, current_metric)

        if epochs_since_improvement == config.training.early_stopping_epochs:
            print('Training has been interrupted by early stopping.')
            break
        
        torch.cuda.empty_cache()
        gc.collect()

    if config.data.test_size > 0.0:
        test(config, model, test_loader, loss_function)

    if config.training.verbose_plots:
        draw_plots(train_losses, val_losses, metrics, learning_rates)
        


def save_model(config, model, epoch, train_loss, val_loss, metric, optimizer, epochs_since_improvement, name, scheduler, image_size, scaler):
    '''Save PyTorch model.'''

    torch.save({
        'model': model.state_dict(),
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'metric': metric,
        'optimizer': optimizer.state_dict(),
        'epochs_since_improvement': epochs_since_improvement,
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict(),
        'image_size': image_size,
    }, os.path.join(config.paths.path_to_checkpoints, name))


def draw_plots(train_losses, val_losses, metrics, lr_changes):
    '''Draw plots of losses, metrics and learning rate changes.'''

    # Learning rate changes
    plt.plot(range(len(lr_changes)), lr_changes, label='Learning Rate')
    plt.legend()
    plt.title('Learning rate changes')
    plt.show()

    # Validation and train losses
    plt.plot(range(len(train_losses)), train_losses, label='Train Loss')
    plt.plot(range(len(val_losses)), val_losses, label='Validation Loss')
    plt.legend()
    plt.title('Changes of validation and train losses')
    plt.show()

    # Metric changes
    plt.plot(range(len(metrics)), metrics, label='Metric')
    plt.legend()
    plt.title('Metric changes')
    plt.show()


def print_report(t, train_loss, val_loss, metric, best_metric):
    '''Print report of one epoch.'''

    print(f'Time: {t} s')
    print(f'Train Loss: {train_loss:.4f}')
    print(f'Val Loss: {val_loss:.4f}')
    print(f'Current Metric: {metric:.4f}')
    print(f'Best Metric: {best_metric:.4f}')


def save_log(path, epoch, train_loss, val_loss, best_metric):
    '''Save log of one epoch.'''

    with open(path, 'a') as file:
        file.write('epoch: ' + str(epoch) + ' train_loss: ' + str(round(train_loss, 5)) + 
                   ' val_loss: ' + str(round(val_loss, 5)) + ' best_metric: ' + 
                   str(round(best_metric, 5)) + '\n')
