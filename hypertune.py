from src.data import datasets
from src.models.rnn_models import GRUmodel
from src.models.metrics import Accuracy
from src.models import train_model
from src.settings import SearchSpace, TrainerSettings, presets
from pathlib import Path
from ray.tune import JupyterNotebookReporter
from ray import tune
import torch
import ray
from typing import Dict
from ray.tune import CLIReporter
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB



def train(config: Dict, checkpoint_dir=None):
    """
    The train function should receive a config file, which is a Dict
    ray will modify the values inside the config before it is passed to the train
    function.
    """
    
    trainloader, validloader = datasets.get_arabic(presets)

    accuracy = Accuracy()
    model = GRUmodel(config)

    trainersettings = TrainerSettings(
        epochs=50,
        metrics=[accuracy],
        logdir="modellog",
        train_steps=len(trainloader),
        valid_steps=len(validloader),
        tunewriter=["ray"],
        scheduler_kwargs={"factor": 0.5, "patience": 5},
        earlystop_kwargs=None,
    )

    trainer = train_model.Trainer(
        model=model,
        settings=trainersettings,
        loss_fn=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam,
        traindataloader=trainloader,
        validdataloader=validloader,
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
    )
    trainer.loop()


if __name__ == "__main__":
    ray.init()

    config = SearchSpace(
        input_size=13,
        output_size=20,
        tune_dir=Path("models/ray").resolve(),
        data_dir=Path("data/raw").resolve(),
    )
    reporter = CLIReporter()
    reporter.add_metric_column("Accuracy")

    bohb_hyperband = HyperBandForBOHB(
        time_attr="training_iteration",
        max_t=50,
        reduction_factor=3,
        stop_last_trials=False,
    )

    bohb_search = TuneBOHB()
    analysis = tune.run(
        train,
        config=config.dict(),
        metric="test_loss",
        mode="min",
        progress_reporter=reporter,
        local_dir=config.tune_dir,
        num_samples=50,
        search_alg=bohb_search,
        scheduler=bohb_hyperband,
        verbose=1,
    )

    ray.shutdown()
