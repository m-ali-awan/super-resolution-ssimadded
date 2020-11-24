import torch
import torchvision
import os
from PIL import Image
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torch.nn as nn
import copy
import numpy as np




def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()
    
    
    
def crappify(directory_path,path_to_save):
    
    imgs_to_crappify=os.listdir(directory_path)
    i=0
    for filename in imgs_to_crappify:

        if filename.endswith('.jpg'):
            img=Image.open(os.path.join(directory_path,filename))
            img_r=img.resize(size=(60,70))
            img_r.save('{}/{}'.format(path_to_save,filename))
            i+=1
            
        else:
            print(filename)
            
            
            

def tensor_img_save(tensor_img,path_to_save):
    
    out=tensor_img
    

    img = out.detach().cpu().numpy()[0]
    #convert image back to Height,Width,Channels
    img = np.transpose(img, (1,2,0))

    arr=np.asarray(img)

    imgg=Image.fromarray((arr*255).astype('uint8'))

    imgg.save(path_to_save)
    
    