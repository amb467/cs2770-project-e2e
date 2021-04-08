import argparse, pathlib, pickle, nltk
from collections import Counter

class Vocabulary(object):
    """Simple vocabulary wrapper."""
    
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
                                
    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def tokenize(string):
    return nltk.tokenize.word_tokenize(str(string).lower())   
                 
def build_vocab(data_file, data_set, vocab_path=None, threshold=5):

    if data_set == 'vqa':
        q_row = 4
    elif data_set == 'vqg':
        q_row = 5
    else:
        raise Exception(f'VQGDataset.init: data_set must be vqa or vqg, found {data_set}')
    
    counter = Counter()
        
    with open(data_file) as f:
        for line in f.readlines()[1:]:
            row_data = line.split('\t')
            questions = row_data[q_row].split('---')
            
            for question in questions:
                tokens = tokenize(question)
                counter.update(tokens)
    
    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>') 
        
    # Add the words to the vocabulary if the word frequency of at least 'threshold'
    [vocab.add_word(word) for word, cnt in counter.items() if cnt >= threshold]

    # Output to pickle file if a path is provided
    if(vocab_path):
        with open(vocab_path, 'wb') as f:
            pickle.dump(vocab, f)
    
    return vocab

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=pathlib.Path, help='path for train data file')
    parser.add_argument('--data_set', type=str, help='"vqa" or "vqg"')
    parser.add_argument('--vocab_path', type=pathlib.Path,  help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=5, help='minimum word count threshold')
    args = parser.parse_args()
    
    vocab = build_vocab(args.data_file, args.data_set, args.vocab_path, args.threshold, )
    print(f'Created vocab with size {len(vocab)}')