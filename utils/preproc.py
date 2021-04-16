import configparser, os, pickle
from model3 import EncoderCNN, DecoderRNN
from torchvision import transforms
from utils.vocab import Vocabulary
from utils.data_loader import get_loader 

def get_transform(crop_size):
    return transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
def proc(args, mode, root_dir, file_name):

    q_data_set = args.data_set
    config_path = os.path.join(root_dir, args.config)

    if not (q_data_set == 'vqa' or q_data_set == 'vqg'):
        raise Exception(f'Usage {file_name} [vqa|vqg]: you provided an invalid question data set: {q_data_set}')
    
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
    params = config[q_data_set]
    c['model_dir'] = os.path.join(root_dir, params['model_dir'])
    vocab_path = os.path.join(root_dir, params['vocab_path'])

    # Image preprocessing, normalization for the pretrained resnet
    transform = get_transform(crop_size)
    
    # Load vocabulary wrapper
    class CustomUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if name == 'Vocabulary':
                from utils.vocab import Vocabulary
                return Vocabulary
            return super().find_class(module, name)
    
    with open(vocab_path, 'rb') as f:
    #    vocab = pickle.load(f)
        vocab = CustomUnpickler(f).load()
        
    c['id_to_word'] = lambda x : vocab.id_to_word(x)
    
    # Build data loader
    data_loader = get_loader(image_dir, data_file_path, q_data_set, vocab, transform, batch_size, True, num_workers)

    # Build the models
    encoder = EncoderCNN(embed_size)
    decoder = DecoderRNN(embed_size, hidden_size, len(vocab), num_layers)
    
    return encoder, decoder, data_loader, c