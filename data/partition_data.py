'''
Author Amanda Buddemeyer
CS 2770 Computer Vision
Spring 2021
Project

The purpose of this script is to divide the entire set of data into train, test, and 
validation sets.  The script buckets the images by category score (# of top five 
categories / # of all categories) and attempts to distribute the images evenly into the
sets based on score.

Once the partitions are made, this script uses the url for each image to download it
into the appropriate folder.  It also outputs data files for each set.

This script will also build a vocab from the training set and save it as a pickle file
'''

import math, random, requests, os, pickle
from collections import defaultdict
from build_vocab import build_vocab

data_file_dir = 'data_files'
vocab_file_dir = 'vocab'

# Open all_data.txt file and extract data
images = defaultdict(list)
with open(os.path.join(data_file_dir,'all_data.txt'), 'r') as f:
    lines = f.readlines()
    header = lines[0]
    for line in lines[1:]:
        row_data = line.split('\t')
        img_id = row_data[0]
        i = math.floor(5*float(row_data[1]))
        img_url = row_data[2]
        questions_vqa = row_data[4].split('---')
        questions_vqg = row_data[5].split('---')
        
        img = {
            'row': line,
            'image-id': img_id,
            'url': img_url
        }
        
        images[i].append(img)

# Divide the data into train, validation, and test sets
data_sets = {
    'train': [],
    'val': [],
    'test': []
}

train_val_proportion = 0.08

for img_list in images.values():
    random.shuffle(img_list)
    train_val_size = math.floor(train_val_proportion * len(img_list))
    
    for i in range(train_val_size):
        data_sets['val'].append(img_list.pop())
        data_sets['test'].append(img_list.pop())
    
    data_sets['train'].extend(img_list)

# Output the data sets

for data_set, data_list in data_sets.items():
    random.shuffle(data_list)
    rows = [img['row'] for img in data_list]
        
    # Output the data files for each data set
    file_name = os.path.join(data_file_dir, f'{data_set}_data.txt')
    with open(file_name, 'w') as f:
        f.write(header)
        f.write(''.join(rows))
        
    # Make a vocabulary if this is the training set
    if data_set == 'train':
    	if not os.path.exists(vocab_file_dir):
    		os.mkdir(vocab_file_dir)
    	vocab = build_vocab(file_name, 'vqa', os.path.join(vocab_file_dir,'vocab_vqa.pkl'))
    	vocab = build_vocab(file_name, 'vqq', os.path.join(vocab_file_dir,'vocab_vqq.pkl'))
	
	# Download the image files
	file_dir = 'f{data_set}/'
	if not os.path.exists(file_dir):
		os.mkdir(file_dir)
		
    for img in data_list:
        img_id = img['image-id']
        img_url = img['url']
        img_file_path = os.path.join(file_dir, img_id)
        r = requests.get(img_url)
        
        with open(img_file_path,'wb') as f:
        	f.write(r.content)