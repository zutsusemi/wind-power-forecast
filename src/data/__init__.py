from torch.utils.data import Dataset
from torch.utils.data.dataset import random_split
from typing import Tuple

def prepare_dataset(dataset: str, PATH, step, out_step) -> Tuple[Dataset, Dataset]:
    """prepare the dataset.
    Args:
        dataset (str): the dataset name.
    Returns:
        trainset and testset
    """
    if dataset == "scada":
        import data.scada as scada
        dataset = scada.load(PATH, step, out_step)
        size = [len(dataset) - len(dataset) // 3000, len(dataset) // 3000]
        train, val = random_split(dataset, size)
    elif dataset == 'kddcup':
        import data.kddcup as kddcup
        dataset = kddcup.load(PATH, step, out_step)
        # size = [len(dataset) - len(dataset) // 100, len(dataset) // 100]
        # train, val = random_split(dataset, size)
        train = dataset
        val  = 0.2 
    elif dataset == 'kddcup_orig':
        import data.kdd_original as kddcup_orig
        dataset = kddcup_orig.load(PATH, step, out_step)
        size = [len(dataset) - len(dataset) // 3000, len(dataset) // 3000]
        train, val = random_split(dataset, size)
    else:
        raise NotImplementedError(f"dataset {dataset} is not implemented.")
    return train, val