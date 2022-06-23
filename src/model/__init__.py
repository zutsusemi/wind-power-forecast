import torch
import torch.nn as nn

# import internal libs
from utils import get_logger

def prepare_model(device: torch.device,
                  model_name: str,
                  dataset: str) -> nn.Module:
    """prepare the random initialized model according to the name.
    Args:
        model_name (str): the model name
        dataset (str): the dataset name

    Return:
        the model
    """
    logger = get_logger(__name__)
    logger.info(f"prepare the {model_name} model for dataset {dataset}")
    if model_name == "lstm" and dataset == "scada":
        import model.lstm as lstm
        model = lstm.LSTM(device, 1, 64, 144, 1).to(device)
    elif model_name == 'seq2seq' and dataset == 'scada':
        import model.lstm as lstm
        model = lstm.Seq2SeqLSTM(1, 64, 1, 144, device).to(device)
    else:
        raise ValueError(f"unknown model name: {model_name}")
    return model