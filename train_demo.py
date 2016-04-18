# This program is a demo for using deep learning algorithm for OCR digit recognition
# Based on the tutorial here: 	http://deeplearning.net/tutorial/mlp.html
# A stripped down version: 		https://gist.github.com/honnibal/6a9e5ef2921c0214eeeb
#
# This version of the demo was tested on Linux

import os
import sys
import time
from os import path
from os.path import isfile, join

import random
import gzip
import cPickle

import time

import neuralnetwork as nn
import numpy
import theano
import spectrogram as sg


def demo_load(train_dir):
    
    train_input = [f for f in os.listdir(train_dir) if isfile(join(train_dir, f))]

    classes   = []
    train_set = []
    test_set  = []
    print 'loading data'
    # Training data
    ize = 1
    for element in train_input:
        
        elem_id = element.split(".")[0].split("_")[0]
        print elem_id

        try:
            index = classes.index(elem_id)
        except:
            index = len(classes)
            classes.append(elem_id)

        stgrm = sg.generate_spectrogram(join(train_dir, element))
        if ize == 3:
            exit(1)
        ize = ize + 1

        train_set.extend(_make_array(stgrm, index))

    train_min_nr = min([len(x[0]) for x in train_set])

    train_data = [x[:train_min_nr] for x in train_set]
    train_data = numpy.random.permutation(train_data)

    test_data  = train_data[:130]
    train_data = train_data[1300:]


    return train_data, test_data, classes


def load_data(train_dir, test_dir, noise_dir):
    
    train_input = [f for f in os.listdir(train_dir) if isfile(join(train_dir, f))]
    test_input  = [f for f in os.listdir(test_dir)  if isfile(join(test_dir, f))]

    classes   = []
    train_set = []
    test_set  = []


    # Training data
    for element in train_input:
        
        elem_id = element.split(".")[0].split("_")[0]

        try:
            index = classes.index(elem_id)
        except:
            index = len(classes)
            classes.append(elem_id)

        stgrm = sg.generate_spectrogram(join(train_dir, element))

        train_set.append(_make_array(stgrm, index))

    # Test data

    for element in test_input:        
        elem_id = element.split(".")[0].split("_")[0]
        test_set.append([join(test_dir,element),elem_id])


    train_min_nr = min([len(x) for x in train_set])

    train_data = []

    for x in train_set:
        indexes    = numpy.random.choice(range(len(x)), train_min_nr, replace=False)
        train_data.extend([x[i] for i in indexes])


    return numpy.random.permutation(train_data), test_set, classes


def _make_array(x, y):
    return zip(
        numpy.asarray(x, dtype=theano.config.floatX),
        numpy.asarray([y]*len(x), dtype='int32')
    )



def main(train_dir='train', test_dir='test', noise_dir='pnoise'):

    print '... loading data'        
    train_set, test_set, classes = load_data(train_dir, test_dir, noise_dir)
    
    women_set = [x for x in test_set if x[1][0]=="F"]
    men_set   = [x for x in test_set if x[1][0]=="M"]
    g = open('o_err.txt','a')
    f = open('m_err.txt','a')
    h = open('f_err.txt','a')
    # train_set, test_set, classes = load_data2()
    
    #train_set, test_set, classes = demo_load("mienk")

    print '... building the model'
    
    n = nn.Net(learning_rate=0.000075, train_data=train_set, classes=classes, L2_reg=0.000001)
    n.add_hidden_layer(1201, 1500)
    n.add_hidden_layer(1500, 1250)
    n.add_hidden_layer(1250, 1000)
    n.add_hidden_layer(1000,  500)


    print '... compiling the model'
    n.compile_model()
    current_error = 1
    error         = 1

    # We train the network 100 times
    # Each time we evaluate the results and write out the error percentage
    for epoch in range(1, 2000):

        if error < current_error:
            current_error = error
            print 'Saving matrix...'
            n.save()
            print 'Save completed...'

        z = time.time()
        # print '... training'            
        print 'Training...'
        for i in xrange(len(train_set)):
            n.train_model(i)
        print 'training took {}'.format(time.time()-z)
        print '... calculating error'            
        # compute zero-one loss on validation set
        z = time.time()
        error = numpy.mean([int(x[1] != n.evaluate(x[0])) for x in women_set])
        h.write("{}. epoch: {}\n".format(epoch, error*100))
        error = numpy.mean([int(x[1] != n.evaluate(x[0])) for x in men_set])
        f.write("{}. epoch: {}\n".format(epoch, error*100))
        error = numpy.mean([int(x[1] != n.evaluate(x[0])) for x in test_set])
        g.write("{}. epoch: {}\n".format(epoch, error*100))

        print 'error calc took {}'.format(time.time() - z)
        print('epoch %i, validation error %f %%' % (epoch, error * 100))


if __name__ == '__main__':
    main()
