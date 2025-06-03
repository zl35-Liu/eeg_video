#!/bin/bash 



#SBATCH --job-name=train_sd           # 作业名
#SBATCH --comment="video sd for eeg2video "    # 作业描述

#SBATCH --partition=L40        # 使用哪个分区

#SBATCH --output=%x_%j.out       # 输出文件
#SBATCH --error=%x_%j.err        # 错误输出文件

#SBATCH --time=0-72:00:00        # 时间限制3days
#SBATCH --nodes=1                # 申请1个节点
#SBATCH --ntasks=1               # 申请2个任务(进程)
#SBATCH --cpus-per-task=1        # 每个任务用1个cpu
#SBATCH --mem-per-cpu=30g        # 每个cpu用10G内存
#SBATCH --gres=gpu:L40:1

#SBATCH --mail-type=ALL
#SBATCH --mail-user=Liu_zl35@163.com

# 打印作业信息（可选，用于调试）
echo "Job started at $(date)"
echo "Running on nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"


source  /share/apps/miniconda3/etc/profile.d/conda.sh  # 加载环境变量
conda activate eegvideo
python sd_hpc.py



echo "Job finished at $(date)"