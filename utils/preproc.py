import configparser, os, pickle
from model3 import EncoderCNN, DecoderRNN
from torchvision import transforms
from utils.vocab import Vocabulary
from utils.data_loader import get_loader 

def proc(args, mode, root_dir, file_name):

	image_data_set = args.data_set
	root_dir = os.path.dirname(os.path.realpath(__file__))
	config_path = os.path.join(root_dir, args.config)

	if not (image_data_set == 'vqa' or image_data_set == 'vqg'):
		raise Exception(f'Usage {file_name} [vqa|vqg]: you provided an invalid image data set: {image_data_set}')
	
	config = configparser.ConfigParser()
	config.read(config_path)
	
	c = {}

	# General config parameters
	params = config['general']
	crop_size = int(params['crop_size'])
	embed_size = int(params['embed_size'])
	hidden_size = int(params['hidden_size'])
	num_layers = int(params['num_layers'])
	batch_size = int(params['batch_size'])
	num_workers = int(params['num_workers'])
	c['learning_rate'] = float(params['learning_rate'])
	c['log_step'] = int(params['log_step'])
	c['num_epochs'] = int(params['num_epochs'])

	# Mode-specific parameters
	params = config[mode]
	image_dir = os.path.join(root_dir, params['image_dir'])
	data_file_path = os.path.join(root_dir, params['data_file_path'])

	# Image data set-specific parameters
	params = config[image_data_set]
	config['model_path'] = os.path.join(root_dir, params['model_path'])
	vocab_path = os.path.join(root_dir, params['vocab_path'])

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
	encoder = EncoderCNN(embed_size)
	decoder = DecoderRNN(embed_size, hidden_size, len(vocab), num_layers)
	
	return encoder, decoder, data_loader, c