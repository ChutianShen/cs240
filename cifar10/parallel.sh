CUDA_VISIBLE_DEVICES="" python cifar10_maxout.py --job_name ps --task_index 0 &
CUDA_VISIBLE_DEVICES=0 python cifar10_maxout.py --job_name worker --task_index 0 &
CUDA_VISIBLE_DEVICES=1 python cifar10_maxout.py --job_name worker --task_index 1 &
CUDA_VISIBLE_DEVICES=2 python cifar10_maxout.py --job_name worker --task_index 2 &