import csv
from itertools import islice

import torch
from torch.utils.data import Dataset


class MNISTData(Dataset):
    """
    Custom pytorch dataset for MNIST, compatible with dataloaders. Allows for selecting
    a subset of the MNIST training data for different tasks.

    Arguments:
        train (bool) : determining whether to select the MNIST train or test data
        data_dir (str) : directory in which the MNIST data .csv are saved
        transform (torhcvision.transforms) : transforms run when fetching data samples
        train_ids (set) : indeces (0-59999) of which MNIST samples to use for training
        resample_stage (bool) : if true, returns samples NOT in the train ids
    """

    def __init__(
        self,
        train,
        data_dir,
        transform=None,
        train_ids=set(list(range(60000))),
        resample_stage=False,
    ):

        self.transform = transform
        self.train_ids = train_ids

        if train:
            csv_path = f"{data_dir}/mnist_train.csv"
        else:
            csv_path = f"{data_dir}/mnist_test.csv"

        with open(csv_path, "r") as f:
            csv_data = csv.reader(f, delimiter=",")
            self.data = []
            for row_i, row in enumerate(islice(csv_data, 1, None)):
                if train and ((row_i not in self.train_ids) ^ resample_stage):
                    continue
                int_row = [int(el) for el in row]
                self.data.append(
                    [int_row[0], [int_row[i : i + 28] for i in range(1, 785, 28)]]
                )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        class_id = self.data[idx][0]
        img = self.data[idx][1]
        img_tensor = torch.tensor(img, dtype=torch.float32)
        img_tensor = img_tensor[None]
        if self.transform:
            img_tensor = self.transform(img_tensor)
        return img_tensor, class_id
