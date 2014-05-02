import csv 
import numpy 
import pdb
from data_source import data_source, move

from rnn_classifier import simple_rnn_classifier
from fnn_classifier import simple_fnn_classifier
from window_fnn_classifier import window_fnn_classifier

from sequence_classifier import sequence_classifier



def main():
  src = data_source()
  #clf = sequence_classifier(src)
  #clf = simple_fnn_classifier(src)
  clf = window_fnn_classifier(src)
  clf.start_training() 

if __name__ == "__main__":
    main()
