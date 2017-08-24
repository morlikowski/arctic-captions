import sys
import os

import pandas as pd
import numpy as np

import nltk
from nltk.tokenize import TreebankWordTokenizer

import scipy
import json
import cPickle
from sklearn.feature_extraction.text import CountVectorizer
import pdb

from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model

from skimage.io import imread #TODO replace usage of skimage with keras.preprocessing

from util import images

import theano
print 'Theano version: ' +  theano.__version__

import keras
print 'Keras version: ' + keras.__version__

TRAIN_SIZE = 6000
TEST_SIZE = 1000

annotation_path = '../data/flickr8k/text/Flickr8k.token.txt'

# TODO Here was a different path before, pointing to preprocessed images.
# What are these preprocessed images? Check if further preprocesing is needed!
flickr_image_path = '../data/flickr8k/images/'

# Load the VGG 19 model and use the feature map of the fourth convolutional layer
# before pooling. See Xu et al. 2016, section 4.3
vgg19 = VGG19(weights='imagenet', include_top=False)
vgg19_conv_layer_output = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv3').output)

# This gives the output for a single image. Can be used as a sanity check
#image_data = image.load_img('../cat.jpg', target_size=(244,244))
#x = image.img_to_array(image_data)
#x = np.expand_dims(x, axis=0)
#x = preprocess_input(x)
#y = vgg19_conv_layer_output.predict(x)
#print y.shape

annotations = pd.read_table(annotation_path, sep='\t', header=None, names=['image', 'caption'])
annotations['image_num'] = annotations['image'].map(lambda x: x.split('#')[1])

annotations['image'] = annotations['image'].map(lambda x: os.path.join(flickr_image_path,x.split('#')[0]))

captions = annotations['caption'].values

with open('captions.txt', 'w') as f:
    for caption in captions:
        f.write(caption)
        f.write('\n')

words = nltk.FreqDist(' '.join(captions).split()).most_common()
wordsDict = {i+2: words[i][0] for i in range(len(words))}

print wordsDict

# vectorizer = CountVectorizer(token_pattern='\\b\\w+\\b').fit(captions)
# dictionary = vectorizer.vocabulary_
# dictionary_series = pd.Series(dictionary.values(), index=dictionary.keys()) + 2
# dictionary = dictionary_series.to_dict()

# # Sort dictionary in descending order
# from collections import OrderedDict
# dictionary = OrderedDict(sorted(dictionary.items(), key=lambda x:x[1], reverse=True))

with open('../data/flickr8k/dictionary.pkl', 'wb') as f:
    cPickle.dump(wordsDict, f)
print "Wrote dictionary."

images = pd.Series(annotations['image'].unique())
image_id_dict = pd.Series(np.array(images.index), index=images)

TRAIN_SIZE = 6000
TEST_SIZE = 1000
DEV_SIZE = len(images) - TRAIN_SIZE - TEST_SIZE

caption_image_id = annotations['image'].map(lambda x: image_id_dict[x]).values
cap = zip(captions, caption_image_id)

# split up into train, test, and dev
all_idx = range(len(images))
np.random.shuffle(all_idx)
train_idx = all_idx[0:TRAIN_SIZE]
train_ext_idx = [i for idx in train_idx for i in xrange(idx*5, (idx*5)+5)]
test_idx = all_idx[TRAIN_SIZE:TRAIN_SIZE+TEST_SIZE]
test_ext_idx = [i for idx in test_idx for i in xrange(idx*5, (idx*5)+5)]
dev_idx = all_idx[TRAIN_SIZE+TEST_SIZE:]
dev_ext_idx = [i for idx in dev_idx for i in xrange(idx*5, (idx*5)+5)]


def preprocess_dataset(images, captions, idx, ext_idx, size, save_path):
    # Select training images and captions
    images_train = images[idx]
    captions_train = captions[ext_idx]
    
    # Reindex the training images
    images_train.index = xrange(size)
    image_id_dict_train = pd.Series(np.array(images_train.index), index=images_train)
    # Create list of image ids corresponding to each caption
    caption_image_id_train = [image_id_dict_train[img] for img in images_train for i in xrange(5)]
    # Create tuples of caption and image id
    cap_train = zip(captions_train, caption_image_id_train)

    # Iterate over all image paths, 100 image paths per iteration
    # TODO: Evaluate if iterating over ranges makes sense - maybe load multiple images at once? Is that faster?
    for start, end in zip(range(0, len(images_train)+100, 100), range(100, len(images_train)+100, 100)):
        image_files = images_train[start:end]
        
        for image_file in image_files:
            image_data = image.load_img(image_file, target_size=(244,244))
            x = image.img_to_array(image_data)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            feat = vgg19_conv_layer_output.predict(x)
            #feat = cnn.get_features(image_list=image_files, layers='conv5_3', layer_sizes=[512,14,14])
            
        if start == 0:
            feat_flatten_list_train = scipy.sparse.csr_matrix(np.array(map(lambda x: x.flatten(), feat)))
        else:
            feat_flatten_list_train = scipy.sparse.vstack([feat_flatten_list_train, scipy.sparse.csr_matrix(np.array(map(lambda x: x.flatten(), feat)))])
    
        print "processing images %d to %d" % (start, end)
    
    # TODO Shouldn't it be possible to write and append this on every iteration, not blocking RAM unnecessarily?
    print "Done, writing to: " + save_path
    with open(save_path, 'wb') as f:
        cPickle.dump(cap_train, f,-1)
        cPickle.dump(feat_flatten_list_train, f)

preprocess_dataset(images, captions, train_idx, train_ext_idx, TRAIN_SIZE, '../data/flickr8k/flicker_8k_align.train.pkl')
preprocess_dataset(images, captions, test_idx, test_ext_idx, TEST_SIZE, '../data/flickr8k/flicker_8k_align.test.pkl')
preprocess_dataset(images, captions, dev_idx, dev_ext_idx, DEV_SIZE, '../data/flickr8k/flicker_8k_align.dev.pkl')

### TRAINING SET
#
## Select training images and captions
#images_train = images[train_idx]
#captions_train = captions[train_ext_idx]
#
## Reindex the training images
#images_train.index = xrange(TRAIN_SIZE)
#image_id_dict_train = pd.Series(np.array(images_train.index), index=images_train)
## Create list of image ids corresponding to each caption
#caption_image_id_train = [image_id_dict_train[img] for img in images_train for i in xrange(5)]
## Create tuples of caption and image id
#cap_train = zip(captions_train, caption_image_id_train)
#
## TODO initalize with proper dimensionality
#feat_flatten_list_train = []
#
## Iterate over all image paths, 100 image paths per iteration
## TODO: Evaluate if iterating over ranges makes sense - perhaps use to load multiple images at once? Is that faster?
#for start, end in zip(range(0, len(images_train)+100, 100), range(100, len(images_train)+100, 100)):
#    image_files = images_train[start:end]
#
#    for image_file in image_files:
#        image_data = image.load_img(image_file, target_size=(244,244))
#        x = image.img_to_array(image_data)
#        x = np.expand_dims(x, axis=0)
#        x = preprocess_input(x)
#        feat = vgg19_conv_layer_output.predict(x)
#        #feat = cnn.get_features(image_list=image_files, layers='conv5_3', layer_sizes=[512,14,14])
#    
#    if start == 0:
#        feat_flatten_list_train = scipy.sparse.csr_matrix(np.array(map(lambda x: x.flatten(), feat)))
#    else:
#        feat_flatten_list_train = scipy.sparse.vstack([feat_flatten_list_train, scipy.sparse.csr_matrix(np.array(map(lambda x: x.flatten(), feat)))])
#
#    print "processing images %d to %d" % (start, end)
#
#with open('data/flickr8k/flicker_8k_align.train.pkl', 'wb') as f:
#    cPickle.dump(cap_train, f,-1)
#    cPickle.dump(feat_flatten_list_train, f)
#    pdb.set_trace()
#
### TEST SET
#
## Select test images and captions
#images_test = images[test_idx]
#captions_test = captions[test_ext_idx]
#
## Reindex the test images
#images_test.index = xrange(TEST_SIZE)
#image_id_dict_test = pd.Series(np.array(images_test.index), index=images_test)
## Create list of image ids corresponding to each caption
#caption_image_id_test = [image_id_dict_test[img] for img in images_test for i in xrange(5)]
## Create tuples of caption and image id
#cap_test = zip(captions_test, caption_image_id_test)
#
#for start, end in zip(range(0, len(images_test)+100, 100), range(100, len(images_test)+100, 100)):
#    image_files = images_test[start:end]
#    feat = cnn.get_features(image_list=image_files, layers='conv5_3', layer_sizes=[512,14,14])
#    if start == 0:
#        feat_flatten_list_test = scipy.sparse.csr_matrix(np.array(map(lambda x: x.flatten(), feat)))
#    else:
#        feat_flatten_list_test = scipy.sparse.vstack([feat_flatten_list_test, scipy.sparse.csr_matrix(np.array(map(lambda x: x.flatten(), feat)))])
#
#    print "processing images %d to %d" % (start, end)
#
#with open('data/flickr8k/flicker_8k_align.test.pkl', 'wb') as f:
#    cPickle.dump(cap_test, f)
#    cPickle.dump(feat_flatten_list_test, f)
#
### DEV SET
#
## Select dev images and captions
#images_dev = images[dev_idx]
#captions_dev = captions[dev_ext_idx]
#
## Reindex the dev images
#images_dev.index = xrange(DEV_SIZE)
#image_id_dict_dev = pd.Series(np.array(images_dev.index), index=images_dev)
## Create list of image ids corresponding to each caption
#caption_image_id_dev = [image_id_dict_dev[img] for img in images_dev for i in xrange(5)]
## Create tuples of caption and image id
#cap_dev = zip(captions_dev, caption_image_id_dev)
#
#for start, end in zip(range(0, len(images_dev)+100, 100), range(100, len(images_dev)+100, 100)):
#    image_files = images_dev[start:end]
#    feat = cnn.get_features(image_list=image_files, layers='conv5_3', layer_sizes=[512,14,14])
#    if start == 0:
#        feat_flatten_list_dev = scipy.sparse.csr_matrix(np.array(map(lambda x: x.flatten(), feat)))
#    else:
#        feat_flatten_list_dev = scipy.sparse.vstack([feat_flatten_list_dev, scipy.sparse.csr_matrix(np.array(map(lambda x: x.flatten(), feat)))])
#
#    print "processing images %d to %d" % (start, end)
#
#with open('data/flickr8k/flicker_8k_align.dev.pkl', 'wb') as f:
#    cPickle.dump(cap_dev, f)
#    cPickle.dump(feat_flatten_list_dev, f)
