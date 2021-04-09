import argparse, copy, nltk, os, pathlib, torch, numpy as np
from utils.preproc import proc
from utils.vocab import Vocabulary

def test(encoder, decoder, data_loader, doOutputQuestions=False):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	c = nltk.translate.bleu_score.SmoothingFunction()
	
	bleu_score = 0.0
	
	for i, (images, questions, lengths) in enumerate(data_loader):
		images = images.to(device)
		feature = encoder(images)
		sampled_ids = decoder.sample(feature)
		sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)
		questions = questions.detach().cpu().numpy()
		references = []
		for item in questions:
			references.append(config['id_to_word'](item))
		generated = id_to_word(sampled_ids)
		
		if doOutputQuestions:
			print(f'{i} of {len(data_loader)}: references: {references} | generated: {generated}')	
	
		bleu_score += nltk.translate.bleu_score.sentence_bleu(references,generated,smoothing_function=c.method7)
	
	return bleu_score / float(len(data_loader))

if __name__ == '__main__':

	# Device configuration
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# Config options
	parser = argparse.ArgumentParser(description='CS2770 Project Eval')
	parser.add_argument('data_set', type=str, help='Eval using "vqa" or "vqg" questions')
	parser.add_argument('--config', type=pathlib.Path, default='config.ini', help='The config file')

	args = parser.parse_args()
	root_dir = os.path.dirname(os.path.realpath(__file__))

	encoder, decoder, data_loader, config = proc(args, 'test', root_dir, 'test.py')

	encoder_path = os.path.join(config['model_dir'], 'best_encoder.pth')
	decoder_path = os.path.join(config['model_dir'], 'best_decoder.pth')
	if not os.path.exists(encoder_path):
		raise Exception(f'Encoder does not exist: {encoder_path}')
	if not os.path.exists(decoder_path):
		raise Exception(f'Decoder does not exist: {decoder_path}')
	
	encoder = encoder.to(device)
	decoder = decoder.to(device)
	encoder.load_state_dict(torch.load(encoder_path))
	decoder.load_state_dict(torch.load(decoder_path))

	bleu_score = test(encoder, decoder, data_loader)
	print(f'Average bleu score for test set: {bleu_score}')




