import neuralnetwork as nn

import os
from os import path
from os.path import isfile, join

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

print 'Creating the network'
n = nn.Net()
n.load('matrix.txt')

train_input = [join("test",f) for f in os.listdir("test") if isfile(join("test", f))]

print 'Evaluation'
for elem in train_input:
	name = n.evaluate(elem)
	print "{}\t===>\t{}\t".format(elem.split('/')[-1], name),
	if name in elem:
		print bcolors.OKGREEN + "OK" + bcolors.ENDC
	else:
		print bcolors.FAIL + "FAIL" + bcolors.ENDC