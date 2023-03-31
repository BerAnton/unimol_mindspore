import os

import mindspore as ms
import mindspore.nn as nn
from mindspore import TimeMonitor, LossMonitor, CheckpointConfig, ModelCheckpoint
from mindspore import ParallelMode
from mindspore.dataset import GeneratorDataset
import mindspore.amp as amp
from mindspore.communication import get_group_size, get_rank, init

from unimol.dataset import create_dataset, molecule_collate_fn
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
            ms.set_context(device_target=device,
                           mode=ms.GRAPH_MODE
                          )
            init()
            ms.set_auto_parallel_context(
                device_num=get_group_size(),
                global_rank=get_rank(),
                parameter_broadcast=True,
                parallel_mode=ParallelMode.DATA_PARALLEL,
                strategy_ckpt_save_file=config.checkpoint_path + "strategy.ckpt"
            )
        else:
            init()
            ms.set_auto_parallel_context(
                device_num=get_device_num(),
                parallel_mode=ParallelMode.DATA_PARALLEL,
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
    # create datasets
    train_dataset = create_dataset(config, is_train=True)
    eval_dataset = create_dataset(config, is_train=False)
    token_vocab = train_dataset.token_vocab.copy()
    if config.run_distribute:
        num_shards = get_group_size()
        shard_id = get_rank()
        train_dataset = GeneratorDataset(train_dataset, column_names=COLUMN_NAMES, 
                                         shuffle=True, num_parallel_workers=config.num_parallel_workers,
                                         python_multiprocessing=config.python_multiprocessing,
                                         max_rowsize=config.max_rowsize,
                                         num_shards=num_shards, shard_id=shard_id)
        eval_dataset = GeneratorDataset(eval_dataset, column_names=COLUMN_NAMES, 
                                        shuffle=False, num_parallel_workers=config.num_parallel_workers,
                                        python_multiprocessing=config.python_multiprocessing,
                                        max_rowsize=config.max_rowsize,
                                        num_shards=num_shards, shard_id=shard_id)
    else:
        train_dataset = GeneratorDataset(train_dataset, column_names=COLUMN_NAMES, 
                                         shuffle=True, num_parallel_workers=config.num_parallel_workers,
                                         python_multiprocessing=config.python_multiprocessing,
                                         max_rowsize=config.max_rowsize)
        eval_dataset = GeneratorDataset(eval_dataset, column_names=COLUMN_NAMES, 
                                        shuffle=False, num_parallel_workers=config.num_parallel_workers,
                                        python_multiprocessing=config.python_multiprocessing,
                                        max_rowsize=config.max_rowsize)
    train_dataset = train_dataset.batch(config.batch_size, per_batch_map=molecule_collate_fn)
    eval_dataset = eval_dataset.batch(config.batch_size, per_batch_map=molecule_collate_fn)
    steps = train_dataset.get_dataset_size() # number of batches in epoch
    epochs = config.epochs
    # define callbacks
    time_cb = TimeMonitor(data_size=steps)
    loss_cb = LossMonitor(per_print_times=1)
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
    learning_rate = nn.WarmUpLR(config.peak_lr, config.warmup_steps)
    opt = nn.Adam(net.trainable_params(), learning_rate, config.beta1,
                  config.beta2, config.eps, weight_decay=config.weight_decay,
                  use_amsgrad=False)
    loss_scale_manager = amp.DynamicLossScaleManager(init_loss_scale=2**15)
    model = ms.Model(net, optimizer=opt, amp_level="O2", loss_scale_manager=loss_scale_manager)
    # train model
    model.train(
        epochs, train_dataset, callbacks=callbacks,
        dataset_sink_mode=True
        )


if __name__ == "__main__":
    train_net()
