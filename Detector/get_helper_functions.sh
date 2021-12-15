#!/bin/bash

vision_files=("utils.py" "transforms.py" "coco_eval.py" "engine.py" "coco_utils.py")

for file in ${vision_files[*]} 
do
	if [ ! -e $file ] 
	then
		curl -O -s https://raw.githubusercontent.com/pytorch/vision/main/references/detection/${file}
	fi
done
