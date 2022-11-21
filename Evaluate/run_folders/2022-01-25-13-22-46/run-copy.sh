#!/bin/bash

run_file=run.sh

cuda_num=0,3,4,5

mkdir run_folders
run_folder=run_folders/$(date +20%y-%m-%d-%H-%M-%S)
#run_file=`basename "$0"`



bash ./copyFiles.sh $run_folder $run_file

cd $run_folder && bash ./run.sh $cuda_num $(date +20%y-%m-%d-%H-%M-%S)
