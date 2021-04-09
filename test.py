import argparse, copy, nltk, os, torch, numpy as np
from utils.preproc import proc
from utils.vocab import Vocabulary, id_to_word

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Config options
parser = argparse.ArgumentParser(description='CS2770 Project Eval')
parser.add_argument('data_set', type=str, help='Eval using "vqa" or "vqg" questions')
parser.add_argument('--config', type=pathlib.Path, default='config.ini', help='The config file')

args = parser.parse_args()
root_dir = os.path.dirname(os.path.realpath(__file__))

def test(encoder, decoder, data_loader):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	c = nltk.translate.bleu_score.SmoothingFunction()
	
	bleu_score = 0.0
	
	for images, questions, lengths in data_loader:
		images = images.to(device)
		feature = encoder(images)
		sampled_ids = decoder.sample(feature)
		sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)
		questions = questions.detach().cpu().numpy()
		references = []
		for question in questions:
			references.append(id_to_word(question))
		gen_cap = id_to_word(sampled_ids)
	
		bleu_score += nltk.translate.bleu_score.sentence_bleu(references,gen_cap,smoothing_function=c.method7)
	
	return bleu_score / float(len(data_loader))

encoder, decoder, data_loader, config = proc(args, 'test', root_dir, 'test.py')

encoder = encoder.to(device)
decoder = decoder.to(device)
		
encoder_path = os.path.join(config['model_path'], 'best_encoder.pth')
decoder_path = os.path.join(config['model_path'], 'best_decoder.pth')

encoder.load_state_dict(torch.load(encoder_path))
decoder.load_state_dict(torch.load(decoder_path))

bleu_score = test(encoder, decoder, data_loader)
print(f'Average bleu score for test set: {bleu_score}')




