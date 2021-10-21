import numpy as np  

import os
import gc
import time
from tqdm import tqdm

import torch

import torch.nn.functional as F
from torch.cuda import amp

from data import get_loaders

from utils import *


def train(config, student, teacher, hard_loss, soft_loss, train_loader, optimizer, scheduler, epoch, scaler):
    '''Train loop.'''

    print('Training')

    student.train()
    teacher.eval()

    lambda_ = config.loss.lambda_
    temperature = config.loss.temperature

    if config.student_model.freeze_batchnorms:
        for name, child in (student.named_children()):
            if name.find('BatchNorm') != -1:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True
    
    total_loss = 0.0

    for step, batch in enumerate(tqdm(train_loader)):
        inputs, labels = batch
        inputs, labels = inputs.to(config.training.device), labels.squeeze(1).to(config.training.device)

        if not config.training.gradient_accumulation:
            optimizer.zero_grad()
        
        if config.training.mixed_precision:
            with amp.autocast():

                with torch.no_grad():
                    logits_t = teacher(inputs)

                logits_s = student(inputs)

                hard = hard_loss(F.log_softmax(logits_s/temperature, dim=-1),
                                labels)
                soft = soft_loss(F.log_softmax(logits_s/temperature, dim=-1),
                                F.softmax(logits_t/temperature, dim=-1))
                loss = lambda_ * hard + (1 - lambda_) * soft

                if config.training.gradient_accumulation:
                    loss = loss / config.training.gradient_accumulation_steps
        else:
            with torch.no_grad():
                logits_t = teacher(inputs)

            logits_s = student(inputs)

            loss_gt = hard_loss(F.log_softmax(logits_s/temperature, dim=-1),
                            labels)
            loss_temp = soft_loss(F.log_softmax(logits_s/temperature, dim=-1),
                            F.softmax(logits_t/temperature, dim=-1))
            loss = lambda_ * loss_gt + (1 - lambda_) * loss_temp

        total_loss += loss.item()

        if config.training.mixed_precision:
            scaler.scale(loss).backward()
        else:
            loss.backward()

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
    
    if config.training.warmup_scheduler:
        if epoch < config.training.warmup_epochs:
            scheduler.step()
        else:
            if config.scheduler.interval == 'epoch':
                scheduler.step()
    else:
        if config.scheduler.interval == 'epoch':
            scheduler.step()

    print('Learning rate:', optimizer.param_groups[0]['lr'])

    return total_loss / len(train_loader)


def validation(config, model, val_loader, loss_function):
    '''Validation loop.'''

    print('Validating')

    model.eval()

    total_loss = 0.0
    activation = torch.nn.Softmax(dim=1)
    preds, targets, conf = [], [], []

    with torch.no_grad():
        for step, batch in enumerate(tqdm(val_loader)):
            inputs, labels = batch
            inputs, labels = inputs.to(config.training.device), labels.squeeze(1).to(config.training.device)

            outputs = model(inputs) # inputs.float()???

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

            outputs = model(inputs) # inputs.float()???

            preds.append(outputs.sigmoid().to('cpu').numpy())
            targets.append(labels.to('cpu').numpy())

            loss = loss_function(outputs, labels)
            total_loss += loss.item()

    metric = get_metric(np.concatenate(targets), np.concatenate(preds))
    print('Test Loss:', total_loss / len(test_loader), '\nTest Metric:', metric)


def kd_training(config):
    '''The main function.'''

    # Create working directory
    if not os.path.exists(config.paths.path_to_checkpoints):
        os.makedirs(config.paths.path_to_checkpoints)

    if config.data.test_size == 0.0:
        train_loader, val_loader = get_loaders(config)
    else:
        train_loader, val_loader, test_loader = get_loaders(config)

    torch.cuda.empty_cache()

    # Get objects
    # model = get_model(config, type="student")
    # teacher = get_model(config, type="teacher")
    model = Net(config.student_model.params, pretrained=config.student_model.pretrained).to(config.training.device)
    teacher = Net(config.teacher_model.params, pretrained=config.teacher_model.pretrained).to(config.training.device)
    teacher.load_state_dict(torch.load(config.teacher_model.checkpoint)["model"])
    teacher.eval()

    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)

    validation_loss = get_loss(config, type="validation")
    hard_loss = get_loss(config, type="hard")
    soft_loss = get_loss(config, type="soft")

    if config.training.mixed_precision:
        scaler = amp.GradScaler()
    else:
        scaler = None

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
    
    # Store transforms to use them after warmup stage
    transforms = config.augmentations.transforms
    image_size = config.data.start_size

    if config.resume_from_checkpoint.last_epoch > 0:
        current_epoch = config.resume_from_checkpoint.last_epoch
    else:
        current_epoch = 0

    # Training
    for epoch in range(current_epoch, config.training.num_epochs):
        print('\nEpoch: ' + str(epoch + 1))

        if epoch in [0, 1, 2]:
            config.data.cutmix = False
            config.data.mixup = False
            config.data.fmix = False
            config.data.pre_transforms = [
                {
                    'name': 'Resize',
                    'params': {
                        'height': 224,
                        'width': 224,
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
            config.data.cutmix = False
            config.data.mixup = False
            config.data.fmix = False
            config.data.pre_transforms = [
                {
                    'name': 'Resize',
                    'params': {
                        'height': 256,
                        'width': 256,
                        'p': 1.0
                    }
                },
            ]

            config.augmentations.transforms = [
                {
                    'name': 'RandomCrop',
                    'params': {
                        'height': 224,
                        'width': 224,
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
            config.data.cutmix = False
            config.data.mixup = False
            config.data.fmix = False
            config.data.pre_transforms = [
                {
                    'name': 'Resize',
                    'params': {
                        'height': 256,
                        'width': 256,
                        'p': 1.0
                    }
                },
            ]

            config.augmentations.transforms = [
                {
                    'name': 'RandomCrop',
                    'params': {
                        'height': 224,
                        'width': 224,
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
            config.data.cutmix = True
            config.data.mixup = True
            config.data.fmix = True
            config.data.pre_transforms = [
                {
                    'name': 'Resize',
                    'params': {
                        'height': 256,
                        'width': 256,
                        'p': 1.0
                    }
                },
            ]

            config.augmentations.transforms = [
                {
                    'name': 'RandomCrop',
                    'params': {
                        'height': 224,
                        'width': 224,
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
            config.data.cutmix = False
            config.data.mixup = False
            config.data.fmix = False
            config.data.pre_transforms = [
                {
                    'name': 'Resize',
                    'params': {
                        'height': 256,
                        'width': 256,
                        'p': 1.0
                    }
                },
            ]

            config.augmentations.transforms = [
                {
                    'name': 'RandomCrop',
                    'params': {
                        'height': 224,
                        'width': 224,
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


        # Applying progressive resizing
        # if image_size < config.data.final_size and epoch > config.training.warmup_epochs:
        #     image_size += config.data.size_step
        #     config.augmentations.pre_transforms = [
        #         {
        #             'name': 'Resize',
        #             'params': {
        #                 'height': image_size,
        #                 'width': image_size,
        #                 'p': 1.0
        #             }
        #         }
        #     ]
        
        # No transforms for warmup stage
        # if epoch < config.training.warmup_epochs:
        #     config.augmentations.transforms = [
        #         {
        #             'name': 'HorizontalFlip',
        #             'params': {
        #                 'p': 0.5
        #             }
        #         }
        #     ]
        # else:
        #     config.augmentations.transforms = transforms

        print('Image size: ' + str(image_size))

        start_time = time.time()

        train_loss = train(config, model, teacher, hard_loss, soft_loss, 
                            train_loader, optimizer, scheduler, epoch, scaler)
        val_loss, predictions, targets, conf = validation(config, model, val_loader, validation_loss)
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
                       'best.pt', scheduler, image_size)
        else:
            epochs_since_improvement += 1

        if epoch % config.training.save_step == 0:
            save_model(config, model, epoch, train_loss, val_loss, current_metric, optimizer, epochs_since_improvement,
                       f'{epoch + 1}_epoch.pt', scheduler, image_size)

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
        test(config, model, test_loader, validation_loss)

    if config.training.verbose_plots:
        draw_plots(train_losses, val_losses, metrics, learning_rates)