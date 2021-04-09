import argparse, copy, os, pathlib, torch
from test import test
from utils.preproc import proc
from utils.vocab import Vocabulary

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Config options
parser = argparse.ArgumentParser(description='CS2770 Project Eval')
parser.add_argument('data_set', type=str, help='Eval using "vqa" or "vqg" questions')
parser.add_argument('--config', type=pathlib.Path, default='config.ini', help='The config file')

args = parser.parse_args()
root_dir = os.path.dirname(os.path.realpath(__file__))

encoder, decoder, data_loader, config = proc(args, 'val', root_dir, 'validate.py')

# Make sure that models exist that we are validating
encoder_path = {}
decoder_path = {}

for epoch in range(1,config['num_epochs']+1):
	encoder_path[epoch] = os.path.join(config['model_dir'], f'encoder-{epoch}.pth')
	decoder_path[epoch] = os.path.join(config['model_dir'], f'decoder-{epoch}.pth')
	if not os.path.exists(encoder_path[epoch]):
		raise Exception(f'Encoder does not exist: {encoder_path[epoch]}')
	if not os.path.exists(decoder_path[epoch]):
		raise Exception(f'Decoder does not exist: {decoder_path[epoch]}')
	

# Put models on device
encoder = encoder.to(device)
decoder = decoder.to(device)

# Validate
best_bleu_score = 0
best_encoder_path = os.path.join(config['model_dir'], 'best_encoder.pth')
best_decoder_path = os.path.join(config['model_dir'], 'best_decoder.pth')

for epoch in range(1,config['num_epochs']+1):	
	encoder.load_state_dict(torch.load(encoder_path[epoch]))
	decoder.load_state_dict(torch.load(decoder_path[epoch]))
	
	bleu_score = test(encoder, decoder, data_loader, config['id_to_word'])
	
	if bleu_score > best_bleu_score:
		best_bleu_score = bleu_score
		best_encoder = copy.deepcopy(encoder.state_dict())
		best_decoder = copy.deepcopy(encoder.state_dict())
		torch.save(best_encoder, best_encoder_path)
		torch.save(best_decoder, best_decoder_path)