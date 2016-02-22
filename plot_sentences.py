# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 16:30:32 2016

@author: hedi
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def get_colors(num_colors):
    import colorsys
    colors=[]
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i/360.
        lightness = (50 + np.random.rand() * 10)/100.
        saturation = (90 + np.random.rand() * 10)/100.
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return colors
    
if __name__=="__main__":
    labels = np.loadtxt('labels').astype('int')
    reduced = np.loadtxt('reduced')
    reduced_w2v = np.loadtxt('reduced_w2v')
    colormap = get_colors(len(set(labels)))
    categories = dict((k,[]) for k in set(labels))
    categories_w2v = dict((k,[]) for k in set(labels))
    for emb, emb_w2v, label in zip(reduced, reduced_w2v,labels):
        categories[label].append(emb)
        categories_w2v[label].append(emb_w2v)
    plt.close('all')
    legends = {0:'civile', 1:'commerciale',2:'criminelle', 3:'JURI', 4:'sociale'}
    fig = plt.figure(figsize=(20,10))
    plt.subplot(121)
    for c in categories:
#        if c == 3 or c==0:
#            continue
        embs = np.array(categories[c])
        plt.scatter(embs[:,0], embs[:,1],  s=15,
                    c=colormap[c], linewidth=0., alpha=0.7, label=legends[c])
    plt.legend()
    plt.title('Skip-thoughts dim_words=310 dim_hidden=1200 n_words=10000 batch_size=128')
    plt.subplot(122)
    for c in categories_w2v:
#        if c == 3 or c==0:
#            continue
        embs = np.array(categories_w2v[c])
        plt.scatter(embs[:,0], embs[:,1], s=15,
                    c=colormap[c], linewidth=0., alpha=0.7, label=legends[c])
    plt.legend()
    plt.title('Mean word2vecs of jurisprudences')
    fig.tight_layout()
    plt.savefig('model1')