from readline import insert_text
from tkinter import image_names
import numpy as np
from PIL import Image
import random
from torchvision.transforms import InterpolationMode
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
import cv2 
import torchvision

# ============================================================= For image
class Compose(object):
    def __init__(self, transforms):
        self.transforms=transforms
    
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image,target

class Resize(object):
    def __init__(self,output_size=384, train=True) -> None:
        self.size=output_size
        self.train=train

    def __call__(self, image, target):
        image = F.resize(image, (self.size, self.size))
        # we must need to test on the original size 
        if self.train:
            target = F.resize(target, (self.size, self.size), interpolation=InterpolationMode.NEAREST)
         
        return image, target

class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = torch.tensor(np.asarray(target), dtype=torch.int64)
        return image, target

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        # print(image.shape, '.,..')
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


# We don't apply other complex data augment
def get_img_transform(size, train_mode=True):
    transforms = []
    transforms.append(Resize(size, train_mode))
    transforms.append(ToTensor())
    transforms.append(Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]))
    return Compose(transforms)



# ============================================================= For video

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
    
def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    
    """
    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
    # if float(torchvision.__version__[:3]) < 0.7:
    #     if input.numel() > 0:
    #         return torch.nn.functional.interpolate(
    #             input, size, scale_factor, mode, align_corners
    #         )

    #     output_shape = _output_size(2, input, size, scale_factor)
    #     output_shape = list(input.shape[:-2]) + list(output_shape)
    #     return _new_empty_tensor(input, output_shape)
    # else:
    #     return torchvision.ops.misc.interpolate(input, size, scale_factor, mode, align_corners)
    return torchvision.ops.misc.interpolate(input, size, scale_factor, mode, align_corners)


class Vid_Resize(object):
    def __init__(self,output_size=384, train=True) -> None:
        self.size=output_size
        self.train=train

    def __call__(self, clip, target):
        rescaled_image = []
        for image in clip:
            rescaled_image.append(F.resize(image, (self.size, self.size)))
        
        # we must need to test on the original size 
        if self.train:
            rescaled_target = target.copy()
            # breakpoint()
            rescaled_target['masks'] = F.resize(rescaled_target['masks'], (self.size, self.size), interpolation=InterpolationMode.NEAREST)
            # for target_ in rescaled_target:
            #     target_['masks'] = F.resize(target_['masks'], (self.size, self.size), interpolation=InterpolationMode.NEAREST)
        else:
            rescaled_target = target.copy()
        return rescaled_image, rescaled_target
    
class Vid_ToTensor(object):
    def __call__(self, clip, target):
        img = []
        for im in clip:
            img.append(F.to_tensor(im))
        return img, target
    
    
class Vid_Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, clip, target=None):
        image = []
        for im in clip:
            image.append(F.normalize(im, mean=self.mean, std=self.std))
        if target is None:
            return image, None
        target = target.copy()
        h, w = image[0].shape[-2:]
        # if "boxes" in target:
        #     boxes = target["boxes"]
        #     boxes = box_xyxy_to_cxcywh(boxes)
        #     boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
        #     target["boxes"] = boxes
        return image, target
    
def get_vid_transform(train_mode='val', size=640):
    # normalize = T.Compose([
    #     T.ToTensor(),
    #     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])

    transforms = []
    transforms.append(Vid_Resize(size, train_mode))
    transforms.append(Vid_ToTensor())
    transforms.append(Vid_Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]))
    return Compose(transforms)

    