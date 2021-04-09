# CS2770 Project End-to-End Model

This code is based on <https://github.com/SathwikTejaswi/Neural-Image-Captioning>

## Description
This code trains, validates, and tests an end-to-end model for Visual Question Generation (VQG).

With respect to the project, the final work remains to be done.  There may be additional work that we can do if we have time, but this is the minimum:

* Integrate the interpretable CNN into the encoder.  Since the interpretable CNN is a categorization CNN that works with one category at a time, our project report suggested that we would actually make five CNNs, one for each of the top five categories in the COCO data set, and then combine their output encodings into one minimized vector.  It may also be necessary to simplify that architecture of the interpretable CNN since we will be training five of them in each training run.
* Implement an attention mechanism to go between the encoder and decoder
* Implement code that visualizes intermediate layers of the interpretable CNN so that we can analyze them as part of our results.  I'm thinking that we just randomly sample a subset of the test set for visualization.
* Train, validate, and test versions of the model for the VQA and VQG data set
* Create a slide presentation for 4/22

## Data
The file `data/data_files/all_data.txt` lists all relevant data for 2894 images from the COCO data set.  These 2894 images were selected because they meet two criteria:

1. They each had at least one questions in the VQA data set and at least one question in the VQG data set
2. They each include at least one of the top five categories

| Category     | ID | Image Count |
| -------------|----|-------------|
| Person       | 1  | 69,501      |
| Chair        | 62 | 13,934      |
| Car          | 3  | 13,321      |
| Dining Table | 67 | 12,839      |
| Cup.         | 47 |  9,969      |

### Partitioning Data into Train, Validation, and Test Sets
Everything described in this section is already done and does not need to be done again.

The script `data/partition_data.py` was run to partition the images listed in `data/data_files/all_data.txt` into a training, validation, and test set.  The script does the following:

1. Buckets each image based on its "category score", meaning (# of top five COCO categories for this image) / (# of total COCO categories for this image).  To ensure that scores are somewhat uniform across sets, each image is bucketed into one of five buckets based on its score.
2. Randomly select 8% of the images in each bucket for the validation set and 8% from each bucket for the test set.  The rest are sorted into the training set.
3. Output a data file for each set.  You will find these in `data/data_files/`
4. Create `data/train/`, `data/val/`, and `data/test/` directories if they do not yet exist
5. Download the images from each set into its corresponding data folder
6. Generate vocabulary objects (of the class `Vocabulary` from `utils/vocab.py` for all of the VQA and VQG questions, respectively, from the training set.  These are pickled and saved in the `data/vocab/` directory.

### Data File Columns
All of the data files in the `data/data_files/` directory have a header, but here is a breakdown of the columns.  The columns are tab-delimited.

* **image_id**
* **score**: This is (# of COCO categories from the top five for this image) / (# of all COCO categories for this image).  So a score of 1 means that all of the categories associated with this image are in the top five (as described in the table above).  There are 80 COCO categories, so the score can get pretty low, though it's not 0 for any of the images listed in the data files.
* **url**: The URL location of the image
* **categories**: The categories from the top five that apply to this image.  These are delimited by three dashes ('---')
* **vqa**: The VQA questions associated with this image.  These are delimited by three dashes ('---') 
* **vqg**: The VQG questions associated with this image.  These are delimited by three dashes ('---') 

These files are already parsed in `utils/data_loader.py` though there will likely need to be an alteration to that code to parse out the categories.


### Train, Validation, and Test Set Size

| Set | # of Images | # of VQA Questions | # of VQG Questions |
|-----|-------------|--------------------|--------------------|
| Train | 2,434 | 13,492 | 12,164 |
| Validate | 230 | 1,137 | 1,149 |
| Test | 230 | 1,256 | 1,150 |

## Config

All configuration information can be found in `config.ini` and is arranged so that all scripts use the same configuration options.

The only parameter that is not passed via the configuration file is the specification of which data set (VQA or VQG) to use questions from.  This will be passed on the command line, as in:

```
python3 train.py vqa
python3 val.py vqg
python3 test.py vqa
```
Note that the model directories in `config.ini` are set up so that models from different question data sets are saved in different directories (`models/vqa/` and `models/vqg/`).  Therefore, it is impossible to accidentally pass a model trained in, say, VQA questions to be validated with the VQG data set.

## Python Files

#### Main Scripts

`train.py`, `val.py`, `test.py`
As explained in "Config" above, each of these scripts uses the `config.ini` configuration file for most config options and each must be run from the command line with the name of a question data set (VQA or VQG).  These scripts should be run in order.  `train.py` will generate a bunch of candidate models that will be validated in `val.py`.  `val.py` will save the best model and that will be used by `test.py`

### Other Files

| File                   | Description 	                      |
|------------------------|------------------------------------|
| `model3.py`            | The encoder and decoder models     |
| `utils/data_loader.py` | This builds the custom data loader |
| `utils/vocab.py`       | The Vocabulary object used to build a vocab, plus some helper methods |
| `utils/preproc.py`     | Some common preprocessing steps done by all three main scripts, including reading the config, generating the vocab and data loader, and generating the basic encoder and decoder objects |
| `data/partition_data.py` | A script that prepares training, validation, and test data.  This was already run to generate these sets and does not need to be run again |

## Encoder and Decoder Notes

In the encoder (founds in `model3.py`), I swapped out the inception_v3 CNN that had been used with a resnet18 instance.  Accordingly, I also switched the layer processing in the `forward` function to match the resnet18 layers.  I stopped processing at the penultimate layer because the last one was outputting a 1x1 image that was throwing an error.  Otherwise, I left everything in this file exactly as I found it.



