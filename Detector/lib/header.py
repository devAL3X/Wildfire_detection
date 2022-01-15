import pandas as pd
from PIL import Image
import torch
import transforms as T

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


class WildfireDataset(torch.utils.data.Dataset):
    def __init__(self, df_sample, annotations="_annotations.csv", transforms=None):
            '''
            df_sample : could be 'train', 'test' or 'valid'
            '''
            self.df_sample = df_sample
            self.annotations = annotations
            self.transforms = transforms
            self.path = "data/" + df_sample + "/"
            self.df = pd.read_csv(self.path + annotations, header=None, names=["img", "xmin", "ymin",        
                                                                    "xmax", "ymax", "alpha"])
            self.df.drop(["alpha"], axis=1, inplace=True)
            self.img_names = self.df["img"].to_numpy()
            
    def __getitem__(self, idx):
        # load images and bounding boxes
        img = Image.open(self.path + self.img_names[idx])
        boxes = self.df[self.df["img"] == self.img_names[idx]][self.df.columns[1:]].values
        num_objs = len(boxes)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        labels = torch.ones((num_objs,), dtype=torch.int64) # only one lable

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {} # an array of values that will predict
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
    
    def __len__(self):
        return len(self.img_names)
