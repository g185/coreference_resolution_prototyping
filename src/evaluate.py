import hydra
import torch
from omegaconf import omegaconf

from data.pl_data_modules import BasePLDataModule
from models.pl_modules import BasePLModule
from utils.logging import get_console_logger

logger = get_console_logger()


@torch.no_grad()
def evaluate(conf: omegaconf.DictConfig):
    device = "cpu"
    hydra.utils.log.info("Using {} as device".format(device))

    pl_data_module: BasePLDataModule = hydra.utils.instantiate(
        conf.data.datamodule, _recursive_=False
    )

    pl_data_module.prepare_data()
    pl_data_module.setup("test")

    logger.log(f"Instantiating the Model from {conf.evaluation.checkpoint}")
    model = BasePLModule.load_from_checkpoint(
        conf.evaluation.checkpoint,
        _recursive_=False,
    )
    model.to(device)
    model.eval()

    for batch in pl_data_module.test_dataloader:
        output = model(**batch)
        print(output["pred_dict"]["coreferences_matrix_form"])
        
    # do stuff
    return


@hydra.main(config_path="../conf", config_name="root")
def main(conf: omegaconf.DictConfig):
    evaluate(conf)


if __name__ == "__main__":
    main()
