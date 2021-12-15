#!/bin/bash

files=("utils.py" "transforms.py" "coco_eval.py" "engine.py" "coco_utils.py")

for file in ${files[*]}
do
	curl https://raw.githubusercontent.com/pytorch/vision/main/references/detection/${file}  --output ${file}
done
