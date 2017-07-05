#! /bin/bash
mkdir -p data/flickr8k/
cd data/flickr8k/
wget http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/Flickr8k_Dataset.zip
unzip Flickr8k_Dataset.zip
rm Flickr8k_Dataset.zip
mv Flicker8k_Dataset/ images/
mkdir text/
cd text/
wget http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/Flickr8k_text.zip
unzip Flickr8k_text.zip
rm Flickr8k_text.zip
