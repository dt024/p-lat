**Baseline softmax:**

CUDA_VISIBLE_DEVICES='2,5,6,7’' python -m torch.distributed.launch --master_port 1607 --nproc_per_node=4 --use_env main_train.py --model deit_tiny_patch16_224 --batch-size 64 --data-path /cm/shared/tuannmd/imagenet/train --output_dir /home/tuannmd/checkpoint/imagenet_softmax


**P-lap: 2 heads low p=1.5, 2 heads high p=2**

CUDA_VISIBLE_DEVICES='0,1,2,6’' python -m torch.distributed.launch --master_port 1609 --nproc_per_node=4 --use_env main_train.py --model deit_tiny_plap_patch16_224 --batch-size 64 --data-path /cm/shared/tuannmd/imagenet/train --output_dir /home/tuannmd/checkpoint/imagenet_plap

**P-lap: 2 heads low p=1, 2 heads high p=2.5**

CUDA_VISIBLE_DEVICES='0,1,2,6’' python -m torch.distributed.launch --master_port 1609 --nproc_per_node=4 --use_env main_train.py --model deit_tiny_plap3_patch16_224 --batch-size 64 --data-path /cm/shared/tuannmd/imagenet/train --output_dir /home/tuannmd/checkpoint/imagenet_plap

**P-lap: 2 heads low p=1.5, 2 heads high p=2.5**

CUDA_VISIBLE_DEVICES='0,1,2,6’' python -m torch.distributed.launch --master_port 1609 --nproc_per_node=4 --use_env main_train.py --model deit_tiny_plap4_patch16_224 --batch-size 64 --data-path /cm/shared/tuannmd/imagenet/train --output_dir /home/tuannmd/checkpoint/imagenet_plap
