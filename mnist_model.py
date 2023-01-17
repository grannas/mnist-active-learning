import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms

from data import MNISTData


BATCH_SIZE = 256 if torch.cuda.is_available() else 64


class MNISTModel(LightningModule):
    """
    Extension of the LightningModule defining the model architecture and training.

    Arguments:
        train_ids (set) : indeces (0-59999) of which MNIST samples to use for training
        data_dir (str) : directory where the mnist data .jsons are stored
        train_ratio (float) : share of samples used for training (rest is validation)
        hidden_size (int) : architecture parameter determining width of linear layers
        learning_rate(float) : learning rate used in the training
        verbose (bool) : determines the verbosity of the program
    """

    def __init__(
        self,
        train_ids=set(list(range(60000))),
        data_dir="MNIST",
        train_ratio=0.8,
        hidden_size=64,
        learning_rate=2e-4,
        verbose=False,
    ):
        super().__init__()

        self.verbose = verbose

        self.train_ids = train_ids
        self.data_dir = data_dir

        self.train_ratio = train_ratio
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        self.num_classes = 10
        self.dims = (1, 28, 28)
        channels, width, height = self.dims
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * width * height, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, self.num_classes),
        )

        self.val_accuracy = Accuracy(task="multiclass", num_classes=10)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=10)

    def set_train_set(self, train_ids):
        self.train_ids = train_ids

    def extend_train_set(self, new_train_ids):
        self.train_ids = self.train_ids.union(new_train_ids)

    def run_inference(self, x):
        x = self.model(x)
        return F.softmax(x, dim=1)

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy.update(preds, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_accuracy.update(preds, y)

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_accuracy, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            mnist_full = MNISTData(
                train=True, train_ids=self.train_ids, data_dir=self.data_dir
            )
            print("TRAIN LEN", len(mnist_full))
            train_n = int(self.train_ratio * len(self.train_ids))
            self.mnist_train, self.mnist_val = random_split(
                mnist_full, [train_n, len(self.train_ids) - train_n]
            )

        if stage == "test" or stage is None:
            self.mnist_test = MNISTData(train=False, data_dir=self.data_dir)
            print("TEST LEN", len(self.mnist_test))

        if stage == "resample":
            self.mnist_resample = MNISTData(train=True, data_dir=self.data_dir)
            print("RESAMPLE LEN", len(self.mnist_resample))

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=BATCH_SIZE)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=BATCH_SIZE)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=BATCH_SIZE)
