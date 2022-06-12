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

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_integer("idx", 0, "Image index", short_name="i")
flags.DEFINE_boolean("no_lable", False, "Do not draw lable", short_name="l")
flags.DEFINE_boolean("no_prediction", False, "Do not draw prediction", short_name="p")
flags.DEFINE_boolean("no_accuracy", False, "Do not draw accuracy", short_name="a")

def get_model(num_classes):
    # load an instance segmentation model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new on
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def main(argv):

    model = get_model(num_classes = 2)
    model.load_state_dict(torch.load("models/model.pt", map_location="cpu"))
    
    dataset_test = WildfireDataset("valid", transforms=get_transform(train=False)) 
    
    img, _ = dataset_test[FLAGS.idx]
    label_boxes = np.array(dataset_test[FLAGS.idx][1]["boxes"])

    # put the model in evaluation mode
    model.eval()
    with torch.no_grad():
        prediction = model([img])
    
    image = Image.fromarray(img.mul(255).permute(1, 2,0).byte().numpy())
    draw = ImageDraw.Draw(image)
    
    # draw ground-truth (lables)
    if not FLAGS.no_lable:
        for elem in range(len(label_boxes)):
            draw.rectangle([(label_boxes[elem][0], label_boxes[elem][1]),
            (label_boxes[elem][2], label_boxes[elem][3])],
            outline ="blue", width = 2)
    
    # draw predictions
    if not FLAGS.no_prediction:
        for element in range(len(prediction[0]["boxes"])):
            boxes = prediction[0]["boxes"][element].cpu().numpy()
            score = np.round(prediction[0]["scores"][element].cpu().numpy(),
                            decimals= 4)
            if score > 0.8:
                draw.rectangle([(boxes[0], boxes[1]), (boxes[2], boxes[3])],
                outline ="orange", width = 2)

                if not FLAGS.no_accuracy:
                    draw.text((boxes[0] + 5, boxes[1] + 1), text = str(score), fill=(255,165,0,255))

    image.show()

if __name__ == "__main__":
    app.run(main)
