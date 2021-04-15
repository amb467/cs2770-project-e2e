import argparse, configparser, os, pathlib, random, torch
from model3 import EncoderCNN
from utils.preproc import get_transform
from PIL import Image

class VisualizationDataset():
    def __init__(self, image_count, config, root_dir):
    
        self.img_dir = os.path.join(root_dir, config['test']['image_dir'])
        self.transform = get_transform(int(config['general']['crop_size']))
        
        data_file = os.path.join(root_dir, config['test']['data_file_path'])
        self.images = []

        with open(data_file) as f:
            for line in f.readlines()[1:]:
                row_data = line.split('\t')
                self.images.append(row_data[0])
        
        random.shuffle(self.images)
        self.images = self.images[:image_count]
                    
    def __getitem__(self, index):
        img_id = self.images[index]
        img_file_path = os.path.join(self.img_dir, img_id)
        
        if os.path.exists(img_file_path):         
            image = Image.open(img_file_path).convert('RGB')
        else:
            raise Exception(f'VisualizationDataset.__getitem__: no such image: {img_file_path}')
        
        if self.transform is not None:
            image = self.transform(image)

        return image
        
def get_encoder(config, q_data_set, device, root_dir):
    model_dir = os.path.join(root_dir, config[q_data_set]['model_dir'])
    embed_size = int(config['general']['embed_size'])
    encoder = EncoderCNN(embed_size).to(device)
    encoder_path = os.path.join(model_dir, 'best_encoder.pth')
    
    if not os.path.exists(encoder_path):
        raise Exception(f'Encoder does not exist: {encoder_path}')
    
    encoder.load_state_dict(torch.load(encoder_path))
    
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

    vqa_encoder = get_encoder(config, 'vqa', device, root_dir)
    vqg_encoder = get_encoder(config, 'vqg', device, root_dir)
    
    data_set = VisualizationDataset(args.image_count, config, root_dir)
    
    for i in range(args.image_count):
        image = data_set[i].to(device)
        vqa_feature = vqa_encoder(image)
        print(f'VQA features: {vqa_feature}')
        vqg_feature = vqg_encoder(image)
        print(f'VQG features: {vqg_feature}')
        