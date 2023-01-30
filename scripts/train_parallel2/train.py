import os
import mindspore as ms
import mindspore.nn as nn
from mindspore import TimeMonitor, LossMonitor, CheckpointConfig, ModelCheckpoint, SummaryCollector, Profiler
from mindspore.dataset import GeneratorDataset
from mindspore.communication import get_group_size, get_rank, init
from mindspore.amp import DynamicLossScaleManager

from unimol.dataset import create_dataset, molecule_collate_fn
from unimol.metrics import DistAccuracy
from unimol.model import create_model
from unimol.loss import MoleculePretrainLoss
from unimol.utils import seed_all, config, get_device_num, set_save_ckpt_dir


COLUMN_NAMES = ["masked_atoms", "masked_coords", "masked_distance", "edge_types", 
                "target_atoms", "coordinates", "distance"]


def set_parameters():
    device = config.device_target
    if device == "CPU":
        config.run_distribute = False
    # init context
    if config.mode == "GRAPH":
        ms.set_context(device_target=device, mode=ms.GRAPH_MODE)
    else:
        ms.set_context(device_target=device, mode=ms.PYNATIVE_MODE)
    if config.run_distribute:
        if device == "Ascend":
            init(config.backend_name)
            ms.set_auto_parallel_context(
                device_num=get_group_size(),
                global_rank=get_rank(),
                parameter_broadcast=True,
                parallel_mode="data_parallel",
                strategy_ckpt_save_file=config.checkpoint_path + "strategy.ckpt"
            )
        else:
            init()
            ms.set_auto_parallel_context(
                device_num=get_device_num(),
                parallel_mode="data_parallel",
                gradients_mean=True,
                strategy_ckpt_save_file=config.checkpoint_path + "strategy.ckpt"
            )


def train_net():
    # set parameters
    device = config.device_target
    seed_all(config.seed)
    set_parameters()
    if config.parameter_server:
        model.set_param_ps()
    ckpt_save_dir = set_save_ckpt_dir(config)
    print(os.getcwd())
    # create datasets
    train_dataset = create_dataset(config, is_train=True)
    eval_dataset = create_dataset(config, is_train=False)
    token_vocab = train_dataset.token_vocab.copy()
    train_dataset = GeneratorDataset(train_dataset, column_names=COLUMN_NAMES, 
                                     shuffle=True, num_parallel_workers=config.num_parallel_workers,
                                     python_multiprocessing=config.python_multiprocessing)
    eval_dataset = GeneratorDataset(eval_dataset, column_names=COLUMN_NAMES, 
                                    shuffle=False, num_parallel_workers=config.num_parallel_workers,
                                    python_multiprocessing=config.python_multiprocessing)
    train_dataset = train_dataset.batch(config.batch_size, per_batch_map=molecule_collate_fn)
    eval_dataset = eval_dataset.batch(config.batch_size, per_batch_map=molecule_collate_fn)
    steps = train_dataset.get_dataset_size() # number of batches in epoch
    epochs = config.epochs
    # define callbacks
    time_cb = TimeMonitor(data_size=steps)
    loss_cb = LossMonitor()
    callbacks = [time_cb, loss_cb]   
    if config.save_checkpoint:
        config_ckpt = CheckpointConfig(save_checkpoint_steps=int(config.save_checkpoint_steps * steps),
                                       keep_checkpoint_max=config.keep_checkpoint_max)
        ckpt_cb = ModelCheckpoint(prefix="UniMol", directory=ckpt_save_dir, config=config_ckpt)
        callbacks += [ckpt_cb]
    
    # instantiate model
    loss = MoleculePretrainLoss(token_vocab, config.token_loss_coef, config.coords_loss_coef, 
                                config.distance_loss_coef, config.beta, config.reduction)
    net = create_model(token_vocab, config, loss)
    metrics = {"acc"}
    if config.run_distribute:
        metrics = {"acc": DistAccuracy(batch_size=config.batch_size, device_num=config.device_num)}
    loss_scale = DynamicLossScaleManager()
    learning_rate = nn.WarmUpLR(config.peak_lr, config.warmup_steps)
    opt = nn.Adam(net.trainable_params(), learning_rate, config.beta1, 
                  config.beta2, config.eps, weight_decay=config.weight_decay)
    model = ms.Model(net, optimizer=opt, loss_scale_manager=loss_scale, metrics=metrics,
                     amp_level=config.accuracy_mode, eval_network=net)
    # train model
    model.fit(
        epochs, train_dataset, eval_dataset, callbacks=callbacks,
        valid_dataset_sink_mode=config.dataset_sink_mode,
        dataset_sink_mode=config.dataset_sink_mode,
        #sink_size=train_dataset.get_dataset_size(),
        )


if __name__ == "__main__":
    train_net()
