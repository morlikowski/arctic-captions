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
# There are paths (all for the same file) that end with ".jpg.1". 
# However, there is no corresponding actual image file, leading to an exception.
# Therefore, delete the ".1" part of the paths.
# Is this a known issue with Flickr8k?
sed -i s/\\.1//g Flickr8k.token.txt
