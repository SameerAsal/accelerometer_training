import csv 
import numpy 
import pdb 
import matplotlib as plt
import pdb

#Pybrain related imports:
from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.tools.customxml.networkwriter import NetworkWriter
from pybrain.tools.customxml.networkreader import NetworkReader   
from numpy import array, array_split, apply_along_axis, concatenate, ones, dot, delete, append, zeros, argmax
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection
from pybrain.structure import FeedForwardNetwork

#user;  gender;  age;  how_tall_in_meters;  weight;  body_mass_index;    
#x1;    y1;    z1;    x2;    y2;    z2;x3;y3;z3;x4;y4; z4;        class
#sitting_down= filter(lambda ll: ll[-1].find("sittingdown") != -1 , sensors)

class window_fnn_classifier():
  num_time_steps    = 4
  def __init__(self, ds): 
    self.max_ratio    = 0.0
    self.correct_perc = []

    num_inputs = ds.num_features * window_fnn_classifier.num_time_steps

    self.alldata = ClassificationDataSet(num_inputs,
                                         target = 1, 
                                         nb_classes = ds.get_num_classes(),
                                         class_labels=ds.get_classes())

    for Idx in range(len(ds.all_moves)):
      if not (Idx + window_fnn_classifier.num_time_steps < len(ds.all_moves)):
        continue
  
      features  = []
      for i in range(window_fnn_classifier.num_time_steps):
        features = features + ds.all_moves[Idx + i].get_features()
 
      self.alldata.addSample(features, [ds.get_classes().index(ds.all_moves[Idx].class_)])
               
    print "Number of records: ", len(self.alldata)
    self.tstdata, self.trndata = self.alldata.splitWithProportion(0.25)
    self.trndata._convertToOneOfMany()
    self.tstdata._convertToOneOfMany()

    print "Number of training patterns: ", len(self.trndata)
    print "Input and output dimensions: ", self.trndata.indim, self.trndata.outdim
    print "First sample (input, target, class):"
    print self.trndata['input'][0]
    print self.trndata['target'][0]
    print self.trndata['class'][0]

    print self.trndata['input'][1]
    print self.trndata['target'][1]
    print self.trndata['class'][1]

    in_layer        = LinearLayer(num_inputs)
    hidden_layer_0  = LinearLayer(12)
    hidden_layer_1  = SigmoidLayer(12)
    output_layer    = LinearLayer(self.trndata.outdim)
    
    self.window_fnn = FeedForwardNetwork()

    self.window_fnn.addInputModule(in_layer)
    self.window_fnn.addModule(hidden_layer_0)
    self.window_fnn.addModule(hidden_layer_1)
    self.window_fnn.addOutputModule(output_layer)
  
    #Now add the connections:
    in_to_h0 = FullConnection(in_layer      , hidden_layer_0)
    h0_to_h1 = FullConnection(hidden_layer_0, hidden_layer_1)
    h1_to_out= FullConnection(hidden_layer_1, output_layer)

    self.window_fnn.addConnection(in_to_h0)
    self.window_fnn.addConnection(h0_to_h1)
    self.window_fnn.addConnection(h1_to_out)
    self.window_fnn.sortModules()

    self.trainer = BackpropTrainer(self.window_fnn, dataset=self.trndata, momentum=0.1, verbose=True, weightdecay=0.01)
    self.write_out()

  def start_training(self):
    f = open( "./results/" + str(window_fnn_classifier.num_time_steps) + "window_fnn_perf.txt", "w")
    for i in range(1000):
      print "training step: " , i
      self.trainer.trainEpochs(1)
      err = self.evaluate()
      f.write(str(err) + ",")
      f.flush()
    f.close()

  def evaluate(self):
    print "epoch:" , self.trainer.totalepochs
    correct = 0
    wrong = 0
    self.window_fnn.sortModules()
    for Idx in range (len(self.tstdata)):
      out = self.window_fnn.activate(self.tstdata['input'][Idx])
      if argmax(out) == argmax(self.tstdata['target'][Idx]) : 
        correct += 1
      else:
        wrong += 1 

    correct_ratio = correct*1.0/(wrong + correct)    
    self.correct_perc.append(correct_ratio)

    print "Wrong Predictions: "  , wrong ,   "Ratio = ", wrong*100.0/(wrong+correct) , "%"
    print "Correct Predictions: ", correct,  "Ratio = ", correct*100.0/(wrong+correct) , "%"
    if (self.max_ratio < correct_ratio): 
      print "Found new max, saving network"
      self.write_out("best_perfrming_")
      self.max_ratio = correct_ratio

    return 1 - correct_ratio
 
  def write_out(self, name=""):
    NetworkWriter.writeToFile(self.window_fnn,  "./results/" + name + str(window_fnn_classifier.num_time_steps) +  "_window_fnn.xml")
