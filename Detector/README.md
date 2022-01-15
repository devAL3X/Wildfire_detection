# Detector
Detection of smoke and fire in the image.

## Requirements:
- [Pillow](https://github.com/python-pillow/Pillow/)
- [cython](https://github.com/cython/cython)
- [pytorch](https://github.com/pytorch/pytorch)
- [torchvision](https://github.com/pytorch/vision)

## Dataset

[Wildfire Smoke Dataset](https://public.roboflow.com/object-detection/wildfire-smoke/1) - contains 737 images presplited on train, test and validation samples + bounding boxes (RetinaNet Keras format is used).

Expanding dataset is planned in the future.

## Installing pycocotools
```
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
cp cp -r pycocotools/ ~/wildfire_detection/Detector
```
## Pretrained model

You can download serialised pretrainde model from my [drive](https://drive.google.com/drive/folders/1IrzhmyoD90DNltTF0g3Jr6fyeLT1Ox5Q?usp=sharing) because it is too large for git.