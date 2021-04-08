import torch
import torch.nn as nn
import os, pathlib, pickle, numpy as np
import argparse, configparser
from utils.data_loader import get_loader 
from utils.vocab import Vocabulary
from model3 import EncoderCNN, DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Config options

parser = argparse.ArgumentParser(description='CS2770 Project Train')
parser.add_argument('data_set', type=str, help='Train on "vqa" or "vqg" questions')
parser.add_argument('--config', type=pathlib.Path, default='config.ini', help='The config file')

args = parser.parse_args()
image_data_set = args.data_set
root_dir = os.path.dirname(os.path.realpath(__file__))

if not (image_data_set == 'vqa' or image_data_set == 'vqg'):
	raise Exception(f'Usage train.py [vqa|vqg]: you provided an invalid image data set: {image_data_set}')

config_set = f'train.{image_data_set}'
config = configparser.ConfigParser()
config.read(args.config)
params = config[config_set]
model_path = os.path.join(root_dir, params['model_path'])
crop_size = int(params['crop_size'])
image_dir = os.path.join(root_dir, params['image_dir'])
data_file_path = os.path.join(root_dir, params['data_file_path'])
vocab_path = os.path.join(root_dir, params['vocab_path'])
log_step = int(params['log_step'])
embed_size = int(params['embed_size'])
hidden_size = int(params['hidden_size'])
num_layers = int(params['num_layers'])
num_epochs = int(params['num_epochs'])
batch_size = int(params['batch_size'])
num_workers = int(params['num_workers'])
learning_rate = float(params['learning_rate'])

# Create model directory
if not os.path.exists(model_path):
	os.makedirs(model_path)

# Image preprocessing, normalization for the pretrained resnet
transform = transforms.Compose([
	transforms.Resize(299),
	transforms.RandomCrop(crop_size),
	transforms.RandomHorizontalFlip(), 
	transforms.ToTensor(), 
	transforms.Normalize((0.485, 0.456, 0.406), 
						 (0.229, 0.224, 0.225))])

# Load vocabulary wrapper
with open(vocab_path, 'rb') as f:
	vocab = pickle.load(f)
	
# Build data loader
data_loader = get_loader(image_dir, data_file_path, image_data_set, vocab, transform, batch_size, True, num_workers)

# Build the models
encoder = EncoderCNN(embed_size).to(device)
decoder = DecoderRNN(embed_size, hidden_size, len(vocab), num_layers).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
optimizer = torch.optim.Adam(params, lr=learning_rate)

# Train the models
total_step = len(data_loader)
for epoch in range(num_epochs):
	for i, (images, questions, lengths) in enumerate(data_loader):
		
		# Set mini-batch dataset
		images = images.to(device)
		questions = questions.to(device)
		targets = pack_padded_sequence(questions, lengths, batch_first=True)[0]
		
		# Forward, backward and optimize
		features = encoder(images)
		outputs = decoder(features, questions, lengths)
		loss = criterion(outputs, targets)
		decoder.zero_grad()
		encoder.zero_grad()
		loss.backward()
		optimizer.step()

		# Print log info
		if i % log_step == 0:
			print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
				  .format(epoch, num_epochs, i, total_step, loss.item(), np.exp(loss.item()))) 
		
	torch.save(decoder.state_dict(), os.path.join(model_path, f'decoder-{epoch+1}.pth'))
	torch.save(encoder.state_dict(), os.path.join(model_path, f'encoder-{epoch+1}.pth'))
		
				
