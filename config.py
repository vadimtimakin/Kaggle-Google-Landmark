from omegaconf import OmegaConf

config = {
    'general': {
        'mode': 'classic_training',  # classic_training, kd_training, val_inference, inference, optimize_model
        'experiment_name': 'default',
        'classification_type': 'multiclass', # multiclass or multilabel
        'seed': 0xFACED,
        'num_classes': 81313,
    },
    'paths': {
        'path_to_images': '/train',
        'path_to_csv': 'oversampled.csv',
        'path_to_checkpoints': './checkpoints',
        'path_to_sample_submission': '',
        'path_to_predicted_submission': '',
        'path_to_inference_images': '',
        'log_name': 'log.txt',
        'path_to_weights': ''
    },
    'training': {
        'train_binary_classifier': False,
        'num_epochs': 15,
        'lr': 1e-4, # target learning rate = base lr * warmup_multiplier if warmup_multiplier > 1.0 (0.00001)
        'mixed_precision': True,
        'gradient_accumulation': True,
        'gradient_accumulation_steps': 2,
        'early_stopping_epochs': 10,
        'gradient_clipping': False,
        'clip_value': 1,

        'warmup_scheduler': False,
        'warmup_epochs': 1,
        'warmup_multiplier': 10, # lr needs to be divided by warmup_multiplier if warmup_multiplier > 1.0

        'debug': False,
        'number_of_debug_samples': 1000,
        'device': 'cuda', # cpu / cuda
        'save_step': 1,
        'verbose_plots': False
    },
    'data': {
        'id_column': 'id',
        'target_columns': ['landmark_id'], # list of columns
        'image_format': '.jpg',

        'batch_size': 4, 
        'train_batch_size': '${data.batch_size}',
        'val_batch_size': '${data.batch_size}',
        'test_batch_size': 1,
        'num_workers': 8,

        'kfold': {
            'use_kfold': False,
            'name': 'KFold',
            'group_column': None, # used only for GroupKFold (str or None)
            'current_fold': 0, # 0 - 4
            'params': {
                'n_splits': 5,
                'shuffle': True
            }
        },

        'train_size': 0.8,
        'val_size': 0.2,
        'test_size': 0.0,

        'first_size': 512, 
        'aftercrop_size': 448, 

        # Progressive Resize Parameters
        'start_size': '${data.aftercrop_size}',
        'final_size': '${data.aftercrop_size}',
        'step_size': 32,

        'cutmix': False,
        'mixup': False,
        'fmix': False,
    },
    'augmentations': {
        'pre_transforms': [
            {
                'name': 'Resize',
                'params': {
                    'height': '${data.first_size}',
                    'width': '${data.first_size}',
                    'p': 1.0
                }
            },
        ],
        'transforms': [
            {
                'name': 'RandomCrop',
                'params': {
                    'height': '${data.aftercrop_size}',
                    'width': '${data.aftercrop_size}',
                    'p': 1.0
                }
            },
            {
                "name": "HorizontalFlip",
                "params": {
                    "p": 0.5
                }
            },
        ],
        'post_transforms': [
            {
                'name': 'Resize',
                'params': {
                    'height': '${data.aftercrop_size}',
                    'width': '${data.aftercrop_size}',
                    'p': 1.0
                }
            },
            {
                'name': 'Normalize',
                'params': {
                    'p': 1.0
                }
            },
            {
                'name': '/custom/to_tensor',
                'params': {
                    'p': 1.0
                }
            }
        ]
    },
    'model': {           # Used for the Classic Training
        'name': 'Net',
        'freeze_batchnorms': False,
        'pretrained': True,
        'params': {
            'backbone':'tf_efficientnet_b0',
            'embedding_size': 512,
            'pool': 'gem',
            'p_trainable': True,
            'neck': 'option-D',
            'head':'arc_margin',
            'n_classes': '${general.num_classes}',
            'pretrained_weights': None,
        }
    },
    'student_model': {   # Used for the Knowledge Distillation Training
        'name': 'Net',
        'freeze_batchnorms': False,
        'pretrained': False,
        'params': {
            'backbone':'swsl_resnet18',
            'embedding_size': 512,
            'pool': 'gem',
            'p_trainable': True,
            'neck': 'option-D',
            'head':'arc_margin',
            'n_classes': '${general.num_classes}',
            'pretrained_weights': None,
        }
    },
    'teacher_model': {   # Used for the Knowledge Distillation Training
        'name': 'Net',
        'freeze_batchnorms': False,
        'pretrained': False,
        'params': {
            'backbone':'tf_efficientnet_b0',
            'embedding_size': 512,
            'pool': 'gem',
            'p_trainable': True,
            'neck': 'option-D',
            'head':'arc_margin',
            'n_classes': '${general.num_classes}',
            'pretrained_weights': None,
        },
        'checkpoint': '',
    },
    'optimizer': {
        'name': 'Adam',
        'params': {
            # 'weight_decay':1e-4,
            'lr': '${training.lr}',
        }
    },
    'scheduler': {
        'name': 'CosineAnnealingLR', 
        'interval': 'epoch', # epoch or step
        'params': {
            'T_max': 12,
            'eta_min': 1e-7
        }
    },
    'loss': {
        # Used for the Classic Training
        'classic_loss_name': 'tobeset',
        'classic_loss_params': {
            's': 45,
            'm': 0.4,
            'crit': "bce",
            'class_weights_norm': "batch",
        },

        # Used for the Knowledge Distillation Training
        'lambda_': 0.5,
        'temperature': 1,

        'soft_loss_name': 'KLDivLoss',
        'hard_loss_name': '/custom/ArcFaceLoss',
        'val_loss_name': '/custom/ArcFaceLoss',

        'soft_loss_params': {
            's': 45,
            'm': 0.4,
            'crit': "bce",
            'class_weights_norm': "batch",
        },
        'hard_loss_params': {
            's': 45,
            'm': 0.4,
            'crit': "bce",
            'class_weights_norm': "batch",
        },
        'val_loss_params': {
            's': 45,
            'm': 0.4,
            'crit': "bce",
            'class_weights_norm': "batch",
        },
    },
    'metric': {
        'name': '/custom/gap',
        'mode': 'max',
        'params': {
            # 'average': 'weighted',
            'num_classes': '${general.num_classes}',
        }
    },
    'optimization_params': {
        'fuse_model': False,
        'static_quantization': False,
        'torch_jit_script': False,
        'path_to_load': './checkpoints/best.pt',
        'path_to_save': 'model.pth',
    },
    'inference': {   # Parameters for val_inference and inference modes
        'batch_size': 32,
        'checkpoint': '',
        'device': 'cuda',  # cpu / cuda
        'path_to_images': '',
    },
    'resume_from_checkpoint': {
        'resume': False,
        'optimizer_state': None,
        'scheduler_state': None,
        'epochs_since_improvement': 0,
        'last_epoch': 0
    },
    'logging': {
        'log': False,
        'wandb_username': '',
        'wandb_project_name': ''
    },
}

config = OmegaConf.create(config)
