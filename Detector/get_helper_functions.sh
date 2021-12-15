#!/bin/bash

vision_files=("utils.py" "transforms.py" "coco_eval.py" "engine.py" "coco_utils.py")

for file in ${vision_files[*]} 
do
	if [ ! -e $file ] 
	then
		curl -O -s https://raw.githubusercontent.com/pytorch/vision/main/references/detection/${file}
	fi
done

if [ ! -e "pycocotools" ] 
then
    mkdir pycocotools
fi

cd pycocotools

coco_files=("__init__.py" "_mask.pyx" "coco.py" "cocoeval.py" "mask.py")

for file in ${coco_files[*]} 
do
	if [ ! -e $file ] 
	then
		curl -O -s https://raw.githubusercontent.com/cocodataset/cocoapi/master/PythonAPI/pycocotools/${file}
	fi
done
