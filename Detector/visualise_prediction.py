#!/usr/bin/env python
from ast import arg
import sys
sys.path.insert(1, './lib')

from PIL import Image, ImageDraw
import numpy as np
import torch
import torchvision
from header import WildfireDataset, get_transform

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_model(num_classes):
    # load an instance segmentation model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new on
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def main(*argv) -> None:

    # help arg handle
    if "--help" in argv or "-h" in argv:
        bold_s = "\033[1m"
        bold_e = "\033[0;0m"

        help_msg = (
            "Visualise-prediction\n"
            "Description: visualise lable and prediction on test dataset \n"
            "USAGE: visualise_prediction.py [OPTION] ...\n"
            "-h, --help\t display this message\n"
            "-i\t set index of picture to visualise\n"
            "-l, --no-lable\t do not draw lables\n"
            "-p, --no-prediction\t do not draw prediction\n"
        )

        print(help_msg)
        return 

    model = get_model(num_classes = 2)
    model.load_state_dict(torch.load("models/model.pt"))
    
    dataset_test = WildfireDataset("valid", transforms=get_transform(train=False)) 
    
    # any index from 0 to 137 (size restrictions by the size of test dataset)
    idx = 0 if not "-i" in argv else int(argv[argv.index("-i") + 1]) 

    img, _ = dataset_test[idx]
    label_boxes = np.array(dataset_test[idx][1]["boxes"])

    #put the model in evaluation mode
    model.eval()
    with torch.no_grad():
        prediction = model([img])
    
    image = Image.fromarray(img.mul(255).permute(1, 2,0).byte().numpy())
    draw = ImageDraw.Draw(image)
    
    # draw ground-truth (lables)
    if not ("-l" in argv or "--no-lable" in argv):
        for elem in range(len(label_boxes)):
            draw.rectangle([(label_boxes[elem][0], label_boxes[elem][1]),
            (label_boxes[elem][2], label_boxes[elem][3])],
            outline ="blue", width = 2)
    
    # draw predictions
    if not ("-p" in argv or "--no-prediction" in argv):
        for element in range(len(prediction[0]["boxes"])):
            boxes = prediction[0]["boxes"][element].cpu().numpy()
            score = np.round(prediction[0]["scores"][element].cpu().numpy(),
                            decimals= 4)
            if score > 0.8:
                draw.rectangle([(boxes[0], boxes[1]), (boxes[2], boxes[3])],
                outline ="orange", width = 2)
                draw.text((boxes[0] + 5, boxes[1] + 1), text = str(score), fill=(255,165,0,255))

    image.show()

if __name__ == "__main__":
    main(*sys.argv)
