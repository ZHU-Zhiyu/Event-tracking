# python main.py
cuda_num=$1
run_folder=$2
echo $cuda_num
# CUDA_VISIBLE_DEVICES=$cuda_num python3 Overall.py --envname $run_folder
CUDA_VISIBLE_DEVICES=$cuda_num python3 -m torch.distributed.launch --master_port 3220 --nproc_per_node=4 main.py --envname $run_folder