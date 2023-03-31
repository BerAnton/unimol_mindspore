#!/bin/bash
# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
export GLOG_v=2

get_real_path(){

  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

CONFIG_FILE=$(get_real_path $1)

ulimit -u unlimited
export DEVICE_NUM=8
export RANK_SIZE=8

export SERVER_ID=0
rank_start=$((DEVICE_NUM * SERVER_ID))

rm -rf ./train_parallel_mpi
mkdir ./train_parallel_mpi
cp ../*.py ./train_parallel_mpi
cp *.sh ./train_parallel_mpi
cp -r ../config/* ./train_parallel_mpi
cp -r ../unimol ./train_parallel_mpi
cd ./train_parallel_mpi || exit

echo "start training"
mpirun -v --allow-run-as-root -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout \
       python train.py --config_path=$CONFIG_FILE --output_path './output' &> log &