import argparse, configparser, os, pathlib, random, torch
#import matplotlib.pyplot as plt
import torch.nn.functional as F
from model3 import EncoderCNN
from utils.preproc import get_transform
from PIL import Image
from torchsummary import summary
from torchvision import transforms
#import cv2 as cv 
#from google.colab.patches import cv2_imshow # for image display

class SaveFeatures():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features = output.clone().detach()
    def close(self):
        self.hook.remove()
        
def get_images(image_count, config, root_dir):
    img_dir = os.path.join(root_dir, config['test']['image_dir'])
    transform = get_transform(int(config['general']['crop_size']))
    data_file = os.path.join(root_dir, config['test']['data_file_path'])
    img_ids = []
    images = []

    with open(data_file) as f:
        for line in f.readlines()[1:]:
            row_data = line.split('\t')
            img_ids.append(row_data[0])
        
    random.shuffle(img_ids)
    img_ids = img_ids[:image_count]
    
    for img_id in img_ids:
        img_file_path = os.path.join(img_dir, img_id)
        
        if os.path.exists(img_file_path):         
            image = Image.open(img_file_path).convert('RGB')
        else:
            raise Exception(f'visualize.get_images: no such image: {img_file_path}')
        
        images.append(transform(image))
        
    return img_ids, images
        
def get_encoder(config, q_data_set, root_dir):
    model_dir = os.path.join(root_dir, config[q_data_set]['model_dir'])
    embed_size = int(config['general']['embed_size'])
    encoder = EncoderCNN(embed_size)
    encoder_path = os.path.join(model_dir, 'best_encoder.pth')
    
    if not os.path.exists(encoder_path):
        raise Exception(f'Encoder does not exist: {encoder_path}')
    
    encoder.load_state_dict(torch.load(encoder_path))
    return encoder
    
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
    img_ids, images = get_images(args.image_count, config, root_dir)
    images = [img.to(device) for img in images]
    images = torch.stack(images, 0)
    
    # Get the encoders and create an array of activation objects to capture the features in the convolutional layers
    encoder = {}
    activations = {} 
    capture_layers = [0,1]
    
    for q_data_set in ['vqa', 'vqg']:
        # Make the encoder
        encoder = get_encoder(config, q_data_set, root_dir)
        encoder.to(device)
        encoder.eval()
        #print(f'Summary of model with question data set: {q_data_set}')
        #summary(encode, (3,299,299))
        
        # Set up a hook to capture layer output once the encoder has run on the images
        activations = {layer: SaveFeatures(list(encoder.children())[layer]) for layer in capture_layers}
        
        # Run the encoder on the images
        features = encoder(images)
        
        # Output a visualization of each captured layer
        for layer, activation in activations.items():
            #print(f'For data set {q_data_set} and layer {i}, features: {activation.features}')
            #plt.imshow(activation.features)
            #plt.show()
            
            for img_count, image in enumerate(activation.features):
                image = transforms.ToPILImage(image)
                image_file_name = f'{img_ids[img_count]}_{q_data_set}_{layer}.jpg'
                image_path = os.path.join(out_dir, image_file_name)
                image.save(image_path, 'JPEG')
                
            activation.close()
