import  cPickle as pickle
import numpy as np
import pandas as pd
import nltk
annotation_path = "/tmp3/alvin/arctic_data/coco/captions.token"
annotations = pd.read_table(annotation_path, sep='\t', header=None,names=['image', 'caption'])
captions = annotations['caption'].values
words = nltk.FreqDist(' '.join(captions).split()).most_common()
wordsDict = {words[i][0]:i+2 for i in range(len(words))}
with open('/tmp3/alvin/arctic_data/coco/dictionary.pkl', 'wb') as f:
    pickle.dump(wordsDict, f)

print words[:10]
