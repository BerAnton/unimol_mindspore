# Run single device training

In ./configs/molecule-pretrain.yaml change atoms_vocab_path to absolute path. Then:  
```bash
python train.py --config_path=./config/molecule-pretrain.yaml
```

# Run multi-device training

Due to various restrictions, training could be only run with OpenMPI.  
In ./scripts/run_distribute_train_mpi.sh change 
```bash
export DEVICE_NUM=8
```
to desirable number of devices. Then:  
```bash
bash run_distribute_train_mpi.sh ../config/molecule-pretrain.yaml
```  
Logs will be in ./scripts/train_parallel_mpi.