# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 17:34:45 2015

@author: hedibenyounes
"""

from nltk.tokenize import word_tokenize, sent_tokenize
from gensim.models.word2vec import Word2Vec
from os import listdir
from os import walk
from os.path import join, isfile
from collections import Counter
import sys
import re
fwrite = sys.stdout.write
        
def pretreat(line):
    return [w for w in word_tokenize(line, 'french') if (w.isalnum() or ('-' in w))]

def processLine(line):
    line=re.sub('\s',' ',line) #replace \s by spaces
    line=re.sub("[A-Z]{1}\.\.",'',line)
    line=re.sub('(,|;|"|\(|\)|\.|:|- | -)','',line) #remove punct
    line=re.sub("[\w]{1,5}\'",'',line) #remove apostrophes
    line=line.lower()
    line=line.split()
    return line


class CustomLineSentence(object):
    """Simple format: one sentence = one line; words already preprocessed and separated by whitespace."""
    def __init__(self, all_paths, verbose=False):
        """
        `source` can be either a string or a file object.

        Example::

            sentences = LineSentence('myfile.txt')

        Or for compressed files::

            sentences = LineSentence('compressed_text.txt.bz2')
            sentences = LineSentence('compressed_text.txt.gz')

        """
        self.all_paths = all_paths 
        self.verbose = verbose

    def __iter__(self):
        """Iterate through the lines in the source."""
        N = len(self.all_paths)
        for idx,current_path in enumerate(self.all_paths):
            if self.verbose:
                fwrite('%d/%d\r' % (idx, N))
                sys.stdout.flush()
            for line in open(current_path,'r'):
                L = line.decode('utf-8')
                sentences = re.split('[.][^.]', L)
                for s in sentences:
                    tokenized = processLine(s)
                    if len(tokenized)>2:
                        yield ' '.join(tokenized)
                        
if __name__ == "__main__":
    root_path = '/mnt/data/datasets/legal/jurisprudences/txt'
    all_paths = []
    for path, folders, files in walk(root_path):
        if len(folders) == 0:
            all_paths += [join(path,f) for f in files]
    root_path = '/mnt/data/datasets/legal/legi/txt'
    for path, folders, files in walk(root_path):
        if len(folders) == 0:
            all_paths += [join(path,f) for f in files]
    sentences = CustomLineSentence(all_paths, verbose=True)
    fwrite('%d files retrieved\n' % len(all_paths))
    sys.stdout.flush()
    target_path = "/home/hbenyounes/legal_skipthoughts/lines_for_st.txt"
    with open(target_path,'w') as f:
        for s in sentences:
            f.write(s.encode('utf-8') + '\n')