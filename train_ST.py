import theano
theano.config.floatX = 'float32'
from skipthoughts.training import vocab
from skipthoughts.training import train
import sys
fwrite = sys.stdout.write

if __name__=="__main__":
    data = []
    fwrite('Loading data ...\n')
    for idx,line in enumerate(open('lines_for_st.txt', 'r')):
        fwrite('%2.1f%%\r' % (100.*idx/10655070))
        sys.stdout.flush
        data.append(u'%s' % line.decode('utf-8'))
    fwrite('\nDone\n')
    loc = '/home/ubuntu/legal_skipthoughts/worddict'
    train.trainer(
        data,
        dim_word=620, # word vector dimensionality
        dim=2400, # the number of GRU units
        encoder='gru',
        decoder='gru',
        max_epochs=5,
        dispFreq=1,
        decay_c=0.,
        grad_clip=5.,
        n_words=10000,
        maxlen_w=30,
        optimizer='adam',
        batch_size = 64,
        saveto='/home/ubuntu/legal_skipthoughts/skptght.model',
        dictionary=loc,
        saveFreq=1000,
        reload_=False)