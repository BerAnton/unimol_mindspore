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

CURPATH="$(dirname "$0")"
# shellcheck source=/dev/null
. ${CURPATH}/cache_util.sh

if [ $# != 2 ] && [ $# != 3 ] && [ $# != 4 ]
then
  echo "Usage: bash run_distribute_train.sh [RANK_TABLE_FILE] [CONFIG_PATH] [PRETRAINED_CKPT_PATH](optional)"
  echo "       bash run_distribute_train.sh [RANK_TABLE_FILE] [CONFIG_PATH] "
  exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

PATH1=$(get_real_path $1)
CONFIG_FILE=$(get_real_path $2)

if [ $# == 4 ]
then 
    PRETRAINED_CKPT_PATH=$(get_real_path $4)
fi

if [ ! -f $PATH1 ]
then 
    echo "error: RANK_TABLE_FILE=$PATH1 is not a file"
exit 1
fi 

if [ $# == 4 ] && [ ! -f $PRETRAINED_CKPT_PATH ]
then
    echo "error: PRETRAINED_CKPT_PATH=$PATH3 is not a file"
exit 1
fi

ulimit -u unlimited
export DEVICE_NUM=8
export RANK_SIZE=8
export RANK_TABLE_FILE=$PATH1

export SERVER_ID=0
rank_start=$((DEVICE_NUM * SERVER_ID))

cpus=`cat /proc/cpuinfo| grep "processor"| wc -l`
avg=`expr $cpus \/ $DEVICE_NUM`
gap=`expr $avg \- 1`

for((i=0; i<${DEVICE_NUM}; i++))
do
    start=`expr $i \* $avg`
    end=`expr $start \+ $gap`
    cmdopt=$start"-"$end
    export DEVICE_ID=${i}
    export RANK_ID=$((rank_start + i))
    rm -rf ./train_parallel$i
    mkdir ./train_parallel$i
    cp ../*.py ./train_parallel$i
    cp *.sh ./train_parallel$i
    cp -r ../config/* ./train_parallel$i
    cp -r ../unimol ./train_parallel$i
    cd ./train_parallel$i || exit
    echo "start training for rank $RANK_ID, device $DEVICE_ID"
    env > env.log
    if [ $# == 2 ]
    then    
        taskset -c $cmdopt python train.py --device_num=$RANK_SIZE --config_path=$CONFIG_FILE \
        --output_path './output' &> log &
    fi
    
    if [ $# == 4 ]
    then
        taskset -c $cmdopt python train.py --device_num=$RANK_SIZE --pre_trained=$PRETRAINED_CKPT_PATH \
        --config_path=$CONFIG_FILE --output_path './output' &> log &
    fi
  cd ..
done