import argparse, configparser, math, os, pathlib, random, numpy as np, torch, torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pack_padded_sequence
from utils.preproc import proc
from dataclasses import dataclass
import torch.autograd.variable as Variable
from torchsummary import summary
from PIL import Image
from torchvision import transforms
import torchvision.models as models
#from icnn_resnet_18 import resnet_18
from resnet_18 import resnet_18

def get_density(label):
    if label.shape[1]>1:
        label = torch.from_numpy(label[:,:,0,0])
        density = torch.mean((label>0).float(),0)
    else:
        density = torch.Tensor([0])
    return density
    
class VisualizeImage:

    IMAGE_TO_TENSOR = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.485, 0.456, 0.406), 
                                 (0.229, 0.224, 0.225))])
    
    TENSOR_TO_IMAGE = transforms.ToPILImage()
                                 
    def __init__(self, img_id, img_dir, category, crop_size):
        self.img_id = img_id
        self.img_file_path = os.path.join(img_dir, img_id)
        self.resize_transform = transforms.Resize(crop_size)
        self.category = torch.Tensor([category])
        self.category = torch.unsqueeze(self.category, 1)
        self.category = torch.unsqueeze(self.category, 1)
        self.category = torch.unsqueeze(self.category, 1)
        self.density = get_density(self.category.numpy())
        
        if os.path.exists(self.img_file_path):         
            image = Image.open(self.img_file_path).convert('RGB')
        else:
            raise Exception(f'VisualizeImage.__init__: no such image: {img_file_path}')
        
        self.original_image = self.resize_transform(image)
        self.image = VisualizeImage.IMAGE_TO_TENSOR(self.original_image)
        self.image = self.image.unsqueeze(0)    # Add a batch dimension
        
    def get_images(image_count, config, root_dir):
        img_dir = os.path.join(root_dir, config['test']['image_dir'])
        data_file = os.path.join(root_dir, config['test']['data_file_path'])
        crop_size = int(config['general']['crop_size'])
        img_ids = []
                   
        # Randomly select image_count images from the test set
        with open(data_file) as f:
            for line in f.readlines()[1:]:
                row_data = line.split('\t')
                img_ids.append(row_data[0])
                cat_set = row_data[3]
                cat_set = set([int(i) for i in cat_set.split('---')]) if len(cat_set) > 0 else []
                category = 1 if 1 in cat_set else -1
        
        random.shuffle(img_ids)
        img_ids = img_ids[:image_count]
        
        # Return a list of VisualizeImage objects for each selected image id
        return [VisualizeImage(img_id, img_dir, category, crop_size) for img_id in img_ids]
        
# If the list has length of more than eight, step through to get at most eight
def step_through_list(max_size, lst):
    l = len(lst)    
    if l > max_size:
        step = math.floor(float(l) / float(max_size))
        offset = (l-1) % step
        s = slice(offset, l, step)
        lst = lst[s]
    
    return lst
    
if __name__ == '__main__':

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Config options
    parser = argparse.ArgumentParser(description='CS2770 Project Eval')
    parser.add_argument('--image_count', type=int, default=8, help='The number of images to select randomly from the test set for visualization')
    parser.add_argument('--config', type=pathlib.Path, default='config.ini', help='The config file')
    parser.add_argument('--pretrain_path', type=pathlib.Path, help='Pretrain for ICNN')
    parser.add_argument('--model_dir', type=pathlib.Path, help='Encoder directory')
    parser.add_argument('--out_dir', type=pathlib.Path, default='visualizations', help='The output directory')

    args = parser.parse_args()
    root_dir = os.path.dirname(os.path.realpath(__file__))
    out_dir = os.path.join(root_dir, args.out_dir)
    
    # Create output directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # Read in config
    config_path = os.path.join(root_dir, args.config)
    config = configparser.ConfigParser()
    config.read(config_path)

    # Build the set of images
    img_objs = VisualizeImage.get_images(args.image_count, config, root_dir)
    
    pretrain_path = args.pretrain_path
    encoder = resnet_18(pretrain_path,1,0,'logistic')        #resnet_18(pretrain_path, label_num, dropoutrate, losstype)
    
    for q_data_set in ['vqa', 'vqg']:
        encoder_path = os.path.join(args.model_dir, f'{q_data_set}-encoder-22.pth')
        encoder.load_state_dict(torch.load(encoder_path))
        encoder.to(device)
        encoder.eval()
    
        # Set up a hook to capture layer output once the encoder has run on the images
    
        layers = [9, 11]
        encoder.create_forward_hooks(layers)

        for i, img_obj in enumerate(img_objs):
            #features = encoder(img_obj.image.to(device))
            image = img_obj.image.to(device)
            category = img_obj.category.to(device)
            density = img_obj.density.to(device)
            #features = encoder(Variable(image), category, 1, img_obj.density)
   
            features = encoder(Variable(image), category, torch.Tensor([1]), density)
            plt.figure(figsize=(20, 20))
            pcount = 1
        
            # Output a visualization of each captured layer
            for layer in layers:
                filters = len(list(torch.squeeze(encoder.extract_layer_features(layer))))
                filters = step_through_list(4, list(range(filters)))
            
                for f in filters:
                    image = VisualizeImage.TENSOR_TO_IMAGE(torch.squeeze(encoder.extract_layer_features(layer))[f])
                    image = img_obj.resize_transform(image)
                    plt.subplot(4, 2, pcount)
                    pcount += 1
                    plt.imshow(image)
                    print(f'Plotted image from image id {img_obj.img_id}; set {q_data_set}; layer {layer}; filter {f}')
        
            plt.axis('off')
            plt.savefig(os.path.join(out_dir, f'{q_data_set}_{img_obj.img_id}'))

    encoder.close_forward_hooks()