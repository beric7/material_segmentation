#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH -p v100_normal_q
#SBATCH -A infraeval
module load gcc cmake
module load cuda/9.0.176 
module load cudnn/7.1
module load Anaconda
source activate TF2

cd $PBS_O_WORKDIR
cd ~/COCO-Bridge-2020/MODELS/deeplabv3plus_seg_material/

python main_plus.py -data_directory '/home/beric7/COCO-Bridge-2020/MODELS/deeplabv3plus_seg_material/DATA/Module_3/aug/' \
-exp_directory '/home/beric7/COCO-Bridge-2020/MODELS/deeplabv3plus_seg_material/stored_weights_plus/var_8_aug_40epoch/' \
--epochs 40 --batch 2

exit
