import random

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from torch.distributions import Categorical

from mnist_model import MNISTModel


class MNISTTrainer:
    """
    Basic trainer for the MNIST dataset. Meant to be used as a class template for
    different training regimes.

    Arguments:
        start_samples (int) : how many training samples initially used
        n_iters (int) : how many times the training is repeated
        samples (int) : how many training samples are added training iteration
        data_dir (str) : directory where the mnist data .jsons are stored
    """

    def __init__(self, start_samples, n_iters, samples, data_dir):
        self.start_samples = start_samples
        self.n_iters = n_iters
        self.samples = samples
        self.experiment_name = f"{self.__class__.__name__.lower()}/"

        self.trainer = None
        self.model = MNISTModel(
            train_ids=set(random.sample(range(0, 60000), self.start_samples)),
            data_dir=data_dir,
        )

    def train_model(self, model=None):
        self.trainer = Trainer(
            accelerator="auto",
            devices=1 if torch.cuda.is_available() else None,
            max_epochs=5,
            callbacks=[TQDMProgressBar(refresh_rate=20)],
            logger=CSVLogger(save_dir=f"experiments/{self.experiment_name}"),
        )
        if model is None:
            self.trainer.fit(self.model)
        else:
            self.trainer.fit(model)

    def run_experiment(self):
        self.train_model()
        for _ in range(self.n_iters):
            self.resample_prep()
            self.resample_data()
            self.train_model()
        return self.trainer.test(self.model, ckpt_path="best")

    def resample_data(self):
        self.model.setup(stage="resample")
        scores = []
        for i, data in enumerate(self.model.mnist_resample):
            if i not in self.model.train_ids:
                scores.append(self.calc_score(data[0]))
            else:
                scores.append(-1e10)
        res = np.argpartition(scores, len(scores) - self.samples)[-self.samples :]
        self.model.extend_train_set(set(res))

    def resample_prep(self):
        return

    def calc_score(self, data):
        return 0


class RandomSampling(MNISTTrainer):
    """
    A version of the basic MNIST Trainer where the new training samples are determined
    randomly.

    Arguments:
        start_samples (int) : how many training samples initially used
        data_dir (str) : directory where the mnist data .jsons are stored
    """

    def __init__(self, start_samples, data_dir):
        super().__init__(
            start_samples=start_samples, n_iters=0, samples=0, data_dir=data_dir
        )


class EntropySampling(MNISTTrainer):
    """
    A version of the basic MNIST Trainer which uses entropy to create an active learning
    training loop. The unused data samples for which the trained model yield the highest
    entropy are added to the training samples each training iteration.

    Arguments:
        start_samples (int) : how many training samples initially used
        n_iters (int) : how many times the training is repeated
        samples (int) : how many training samples are added training iteration
        data_dir (str) : directory where the mnist data .jsons are stored
    """

    def __init__(self, start_samples, n_iters, samples, data_dir):
        super().__init__(
            start_samples=start_samples,
            n_iters=n_iters,
            samples=samples,
            data_dir=data_dir,
        )

    def calc_score(self, data):
        preds = self.model.run_inference(data)
        entropy = Categorical(probs=preds).entropy()
        return entropy.item()


class EpistemicSampling(MNISTTrainer):
    """
    A version of the basic MNIST Trainer which uses epistemic uncertainty to create
    an active learning training loop. An ensemble is trained using the same data and
    architecture as the model. The ensemble is used to estimate the epistemic (model)
    uncertainty. The unused data samples for which the ensemble yields the highest
    epistemic uncertainty are added to the training samples each training iteration.

    Arguments:
        start_samples (int) : how many training samples initially used
        n_iters (int) : how many times the training is repeated
        samples (int) : how many training samples are added training iteration
        data_dir (str) : directory where the mnist data .jsons are stored
        ensemble_size (int) : number of members in the ensemble
    """

    def __init__(self, start_samples, n_iters, samples, data_dir, ensemble_size):
        super().__init__(
            start_samples=start_samples,
            n_iters=n_iters,
            samples=samples,
            data_dir=data_dir,
        )
        self.ensemble = [
            MNISTModel(train_ids=self.model.train_ids, data_dir=data_dir)
            for _ in range(ensemble_size)
        ]

    def resample_prep(self):
        for model in self.ensemble:
            model.extend_train_set(self.model.train_ids)
            self.train_model(model)

    def calc_score(self, data):
        preds_l = []
        entropy_l = []
        for model in self.ensemble:
            preds = model.run_inference(data)
            preds_l.append(preds)
            entropy_l.append(Categorical(probs=preds).entropy())
        total_uncertainty = (
            Categorical(probs=torch.mean(torch.stack(preds_l), dim=0)).entropy().item()
        )
        aleatoric_uncertainty = torch.mean(torch.Tensor(entropy_l)).item()
        return total_uncertainty - aleatoric_uncertainty
