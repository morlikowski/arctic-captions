#! /bin/bash
mkdir -p data/flickr8k/
cd data/flickr8k/
wget http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/Flickr8k_Dataset.zip
unzip Flickr8k_Dataset.zip
rm Flickr8k_Dataset.zip
mv Flicker8k_Dataset images
mkdir text/
cd text/
wget http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/Flickr8k_text.zip
unzip Flickr8k_text.zip
rm Flickr8k_text.zip
# There are paths that have no corresponding actual image file, leading to an exception.
# Is this a known issue with Flickr8k?
grep -v "2258277193_586949ec62.jpg" Flickr8k.token.txt > Flickr8k.token.txt.new
mv Flickr8k.token.txt.new Flickr8k.token.txt
