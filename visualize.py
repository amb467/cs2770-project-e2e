import argparse, configparser, math, os, pathlib, random, torch
import matplotlib.pyplot as plt
import torch.nn as nn, torch.nn.functional as F
from model3 import EncoderCNN, SaveFeatures
from utils.preproc import get_transform
from PIL import Image
from torchsummary import summary
from torchvision import transforms
import torchvision.models as models

class VisualizeImage:

    IMAGE_TO_TENSOR = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.485, 0.456, 0.406), 
                                 (0.229, 0.224, 0.225))])
    
    TENSOR_TO_IMAGE = transforms.ToPILImage()
                                 
    def __init__(self, img_id, img_dir, crop_size):
        self.img_id = img_id
        self.img_file_path = os.path.join(img_dir, img_id)
        self.resize_transform = transforms.Resize(crop_size)
        
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
        
        random.shuffle(img_ids)
        img_ids = img_ids[:image_count]
        
        # Return a list of VisualizeImage objects for each selected image id
        return [VisualizeImage(img_id, img_dir, crop_size) for img_id in img_ids]
        
def get_encoder(config, q_data_set, root_dir):
    model_dir = os.path.join(root_dir, config[q_data_set]['model_dir'])
    encoder = EncoderCNN()
    encoder_path = os.path.join(model_dir, 'best_encoder.pth')
    
    if not os.path.exists(encoder_path):
        raise Exception(f'Encoder does not exist: {encoder_path}')
    
    encoder.load_state_dict(torch.load(encoder_path))
    return encoder


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
    
    for q_data_set in ['vqa', 'vqg']:
        # Make the encoder
        encoder = get_encoder(config, q_data_set, root_dir)
        encoder.to(device)
        encoder.eval()
        #print(f'Summary of model with question data set: {q_data_set} and with children: {len(list(encoder.modules()))}')
        #for i, m in enumerate(list(encoder.named_modules())):
        #   print(f'{i}\t{m[0]}')
        #summary(encoder, (3,299,299))
        
        # Set up a hook to capture layer output once the encoder has run on the images
        modules = list(encoder.modules())
        layers = step_through_list(8, [i for i in range(len(modules)) if type(modules[i]) == nn.Conv2d])[-4:]
        activations = {i: SaveFeatures(modules[i]) for i in layers}

        for i, img_obj in enumerate(img_objs):
            features = encoder(img_obj.image.to(device))
            plt.figure(figsize=(20, 20))
            pcount = 1
            
            # Output a visualization of each captured layer
            for layer, activation in activations.items():
                filters = len(list(torch.squeeze(activation.features)))
                filters = step_through_list(4, list(range(filters)))
                
                for f in filters:
                    image = VisualizeImage.TENSOR_TO_IMAGE(torch.squeeze(activation.features)[f])
                    image = img_obj.resize_transform(image)
                    plt.subplot(4, 4, pcount)
                    pcount += 1
                    plt.imshow(image)
                    print(f'Plotted image from image id {img_obj.img_id}; set {q_data_set}; layer {layer}; filter {f}')
            
            plt.axis('off')
            plt.savefig(os.path.join(out_dir, f'{q_data_set}_{img_obj.img_id}'))

        [activation.close() for activation in activations.values()]