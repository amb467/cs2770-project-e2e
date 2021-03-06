import os, torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from utils.vocab import Vocabulary, tokenize

class VQGDataset(data.Dataset):

    CAT_IDS = [1, 62, 3, 67, 47]
    
    def __init__(self, img_dir, data_file, data_set, vocab, transform=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            img_dir: image directory.
            data_file: a tab-delimited file that lists image id, category score, url, categories, vqa questions, and vqg questions
            data_set: either 'vqa' or 'vqg' depending on which questions the loader should deliver
            vocab: the training vocabulary
            transform: image transformer.
        """
        self.img_dir = img_dir
        self.vocab = vocab
        self.transform = transform
        self.categories = []
        self.questions = []
        self.images = []
        self.img_to_url = {}
        
        if data_set == 'vqa':
            q_row = 4
        elif data_set == 'vqg':
            q_row = 5
        else:
            raise Exception(f'VQGDataset.init: data_set must be vqa or vqg, found {data_set}')
        
        with open(data_file) as f:
            for line in f.readlines()[1:]:
                row_data = line.split('\t')
                img_id = row_data[0]
                img_url = row_data[2]
                cat_set = row_data[3]
                cat_set = set([int(i) for i in cat_set.split('---')]) if len(cat_set) > 0 else []
                cat_list = []
                for cat in VQGDataset.CAT_IDS:
                    cat_list.append(1 if cat in cat_set else -1)
                category = cat_list[0]  # We are only doing the Person category right now
                questions = row_data[q_row].split('---')
                self.img_to_url[img_id] = img_url
                
                for question in questions:
                    #self.categories.append(cat_list)
                    self.categories.append(category)
                    self.questions.append(question)
                    self.images.append(img_id)
                 
    def __getitem__(self, index):
        """Returns one data pair (image and question)."""
        category = self.categories[index]
        question = self.questions[index]
        img_id = self.images[index]
        img_file_path = os.path.join(self.img_dir, img_id)
        
        if os.path.exists(img_file_path):         
            image = Image.open(img_file_path).convert('RGB')
        else:
            raise Exception(f'VQGDataset.__getitem__: no such image: {img_file_path}')
        
        if self.transform is not None:
            image = self.transform(image)

        # Convert the question (string) to word ids.
        target = [self.vocab('<start>')]
        target.extend([self.vocab(token) for token in tokenize(question)])
        target.append(self.vocab('<end>'))
        return image, category, torch.Tensor(target)

    def __len__(self):
        return len(self.questions)

def collate_fn(data):
    # Sort a data list by question length (descending order).
    data.sort(key=lambda x: len(x[2]), reverse=True)
    images, categories, questions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)
    
    # Merge categories into a 4D tensor
    categories = torch.Tensor(categories)
    categories = torch.unsqueeze(categories, 1)
    categories = torch.unsqueeze(categories, 1)
    categories = torch.unsqueeze(categories, 1)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(q) for q in questions]
    targets = torch.zeros(len(questions), max(lengths)).long()
    for i, q in enumerate(questions):
        end = lengths[i]
        targets[i, :end] = q[:end]        
    return images, categories, targets, lengths
    
def get_loader(img_dir, data_file, data_set, vocab, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom dataset."""

    vqg_data = VQGDataset(img_dir, data_file, data_set, vocab, transform)

    data_loader = torch.utils.data.DataLoader(dataset=vqg_data, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader
