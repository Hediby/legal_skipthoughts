# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 16:45:12 2016

@author: hedi
"""

import theano
theano.config.floatX = 'float32'
import sys
fwrite = sys.stdout.write
from skipthoughts.training import tools

if __name__=='__main__':
    embed_map = tools.load_googlenews_vectors()
    model = tools.load_model(embed_map)