# python main.py
# CUDA_VISIBLE_DEVICES=$cuda_num python3 Overall.py --envname $run_folder
CUDA_VISIBLE_DEVICES=4 python3 test.py --envname $(date +20%y-%m-%d-%H-%M-%S)| tee $(date +20%y-%m-%d-%H-%M-%S)".txt"
# CUDA_VISIBLE_DEVICES=4 python3 test.py --envname "Test"