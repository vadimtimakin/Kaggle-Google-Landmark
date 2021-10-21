import cv2
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset

from data import get_transforms
from utils import get_model,  calibrate_model
from config import config


class InferenceDataset(Dataset):
    """The Inference Dataset class."""
    def __init__(self, path, transforms=None):
        self.path = path
        self.names = os.listdir(path)
        self.transforms = transforms

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        sample_path = os.path.join(self.path, self.names[idx])

        image = cv2.imread(sample_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transforms:
            image = self.transforms(image=image)['image']

        return image, sample_path


def get_loader_inference(config):
    '''Get loader for the inference'''

    _, transforms = get_transforms(config)

    dataset = InferenceDataset(config.inference.path_to_images, transforms)

    data_loader = DataLoader(dataset, shuffle=False, batch_size=config.inference.batch_size, pin_memory=True, 
                             num_workers=config.data.num_workers, drop_last=False)

    return data_loader

def inference(config, path_to_images, batch_size, device):
    """The inference function."""

    if device is not None:
        config.inference.device = device
    if batch_size is not None:
        config.inference.batch_size = int(batch_size)
    if path_to_images is not None:
        config.inference.path_to_images = path_to_images
    
    model = get_model(config, type='classic')
    model.to(config.inference.device)
    model.eval()

    print("Calibrating...")
    if config.optimization_params.fuse_model:
        model.fuse_model()

    if config.optimization_params.static_quantization and config.inference.device == "cpu":
        model.qconfig = torch.quantization.default_qconfig
        torch.quantization.prepare(model, inplace=True)

        calibrate_model(config, model, num_batches=100)

        torch.quantization.convert(model, inplace=True)

    try:
        model.load_state_dict(torch.load(config.inference.checkpoint)["model"])
    except KeyError:
        model.load_state_dict(torch.load(config.inference.checkpoint))

    if config.optimization_params.torch_jit_script:
        model = torch.jit.script(model)

        calibrate_model(config, model, num_batches=100)

    print("Calibration is finished!")

    torch.cuda.empty_cache()   
    data_loader = get_loader_inference(config)

    preds, paths = [], []
    with torch.no_grad():
        for img, path in tqdm(data_loader):
            
            outputs = model(img.to(config.inference.device))
            preds.append((np.argmax(outputs.sigmoid().to('cpu').numpy()) - 1).tolist())

            paths.append(path[0])

    df = pd.DataFrame()
    df["path"] = paths
    df["label"] = preds
    df.to_csv('submission.csv', index=False)