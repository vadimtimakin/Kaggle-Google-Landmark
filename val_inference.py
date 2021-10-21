import time
import gc
from tqdm import tqdm
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
from icecream import ic

from utils import get_loss, get_model, get_loaders, get_metric, print_size_of_model
from utils import calibrate_model
from config import config

from custom.model import Net


def validation(config, model, val_loader, loss_function):
    '''Validation loop.'''

    print('Validating')

    model.eval()

    total_loss = 0.0
    activation = torch.nn.Softmax(dim=1)
    preds, conf, targets = [], [], []

    predictions, targets = [], []
    with torch.no_grad():
        for step, batch in enumerate(tqdm(val_loader)):
            inputs, labels = batch
            inputs, labels = inputs.to(config.training.device), labels.squeeze(1).to(config.training.device)

            outputs = model(inputs)
            predictions.append(outputs.cpu())
            targets.append(labels.cpu())
            if step  == 700:
                with open("preds.pickle", "wb") as file:
                    pickle.dump(np.concatenate(predictions), file)
                with open("labels.pickle", "wb") as file:
                    pickle.dump(np.concatenate(targets), file)

    return total_loss / len(val_loader), np.concatenate(preds), np.concatenate(targets), np.concatenate(conf)


def val_inference(config):

    config.training.device = config.inference.device
    config.data.val_batch_size = config.inference.batch_size

    if config.data.test_size == 0.0:
        _, val_loader = get_loaders(config)
    else:
        _, val_loader, _ = get_loaders(config)

    torch.cuda.empty_cache()
    gc.collect()

    # Get objects
    if config.model.name == "Net":
        model = Net(config.model.params, pretrained=config.model.pretrained).to(config.training.device)
    else:
        model = get_model(config, type="classic")
    model.to(config.inference.device)
    model.eval()

    print("Calibrating...")
    if config.optimization_params.fuse_model:
        model.fuse_model()

    if config.optimization_params.static_quantization and config.inference.device == "cpu":
        model.qconfig = torch.quantization.default_qconfig
        torch.quantization.prepare(model, inplace=True)

        calibrate_model(config, model, num_batches=8)

        torch.quantization.convert(model, inplace=True)

    try:
        model.load_state_dict(torch.load(config.inference.checkpoint)["model"])
    except KeyError:
        model.load_state_dict(torch.load(config.inference.checkpoint))

    if config.optimization_params.torch_jit_script:
        model = torch.jit.script(model)

        calibrate_model(config, model, num_batches=100)

    print("Calibration is finished!")
    print_size_of_model(model)

    loss_function = get_loss(config, type='classic')

    start_time = time.time()

    val_loss, predictions, targets, conf = validation(config, model, val_loader, loss_function)
    current_metric = get_metric(config, targets, predictions, conf)

    t = int(time.time() - start_time)

    print("Metrics:", round(current_metric, 4))
    print("Val Loss:", round(val_loss, 4))
    print("Time (s):", t)
