import json
import PIL
import numpy as np

def read_class_names(category_names):
    with open(category_names, 'r') as f:
        return json.load(f)
    
def centeredCrop(img, px):
    left_margin = (img.width-px)/2
    bottom_margin = (img.height-px)/2
    right_margin = left_margin + px
    top_margin = bottom_margin + px
    return img.crop((left_margin, bottom_margin, right_margin, top_margin))

def resized(img, px):
    if img.size[0] > img.size[1]:
        img.thumbnail((10000, px))
    else:
        img.thumbnail((px, 10000))
    return img
        
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    mu = (0.485, 0.456, 0.406)
    sigma = (0.229, 0.224, 0.225)
    
    # TODO: Process a PIL image for use in a PyTorch model 
    image = PIL.Image.open(image_path)
    image = resized(image, 255)
    image = centeredCrop(image, 224)  
    
    np_image = np.array(image)
    np_image = np_image/255
    np_image =  (np_image - mu)/sigma   
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image

