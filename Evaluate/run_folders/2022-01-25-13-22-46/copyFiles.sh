#!/bin/bash
#input run_folder, run_file
run_folder=$1
run_file=$2


echo runing folder: $run_folder

mkdir $run_folder

cp ./*.py $run_folder
cp ./*.sh $run_folder
cp ./*.txt $run_folder
cp -r ./util* $run_folder
# cp $run_file $run_folder

# cp -r ./efficientnet_pytorch/ $run_folder
# mkdir ${run_folder}/examples
# mkdir ${run_folder}/examples/imagenet/
# cp -r ./examples/imagenet/main.py ${run_folder}/examples/imagenet/