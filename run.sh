#!/usr/bin/env bash
CUR_DIR=$(cd $(dirname $0); pwd)
#/mnt/bn/video-diffusion
pip3 install triton 
# 取 worker0 第一个 port
ports=(`echo $METIS_WORKER_0_PORT | tr ',' ' '`)
port=${ports[0]}

echo "total workers: ${ARNOLD_WORKER_NUM}"
echo "cur worker id: ${ARNOLD_ID}"
echo "gpus per worker: ${ARNOLD_WORKER_GPU}"
echo "master ip: ${METIS_WORKER_0_HOST}"
echo "master port: ${port}"

export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=${ARNOLD_RDMA_DEVICE}
export NCCL_SOCKET_IFNAME=eth0

ls -l

echo " MASTER_ADDR=${METIS_WORKER_0_HOST} MASTER_PORT=${port} WORLD_SIZE=${ARNOLD_WORKER_NUM} NODE_RANK=${ARNOLD_ID} LOCAL_RAN=0 \
    python3 main.py --devices ${ARNOLD_WORKER_GPU} --num_nodes ${ARNOLD_WORKER_NUM} "$@" "

MASTER_ADDR=${METIS_WORKER_0_HOST} MASTER_PORT=${port} WORLD_SIZE=${ARNOLD_WORKER_NUM} NODE_RANK=${ARNOLD_ID} LOCAL_RAN=0 \
    python3 main.py --devices ${ARNOLD_WORKER_GPU} --num_nodes ${ARNOLD_WORKER_NUM} "$@"