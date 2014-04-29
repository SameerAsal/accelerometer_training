import csv 
import numpy 
import pdb 
import matplotlib as plt
import pdb

#Pybrain related imports:
from pybrain.datasets            import ClassificationDataSet
from pybrain.datasets            import SequenceClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.structure.modules import LSTMLayer
from pybrain.structure.modules import SigmoidLayer
from pybrain.supervised.trainers import RPropMinusTrainer
from pybrain.tools.validation import ModuleValidator
from pybrain.tools.validation import Validator
from numpy import array, array_split, apply_along_axis, concatenate, ones, dot, delete, append, zeros, argmax


#user;  gender;  age;  how_tall_in_meters;  weight;  body_mass_index;    
#x1;    y1;    z1;    x2;    y2;    z2;x3;y3;z3;x4;y4; z4;        class
#sitting_down= filter(lambda ll: ll[-1].find("sittingdown") != -1 , sensors)

class sequence_classifier():
  num_time_steps = 5
  def __init__(self, ds): 

    num_inputs = ds.num_features * sequence_classifier.num_time_steps
    self.alldata = SequenceClassificationDataSet(num_inputs, 
                                                 target = 1,
                                                 nb_classes   = ds.get_num_classes(),
                                                 class_labels = ds.get_classes() )

    for Idx in range(len(ds.all_moves)):
      if not (Idx + sequence_classifier.num_time_steps < len(ds.all_moves)):
        continue
   
      class_first = ds.all_moves[Idx].class_ 
      features  = []
      for i in range(sequence_classifier.num_time_steps):
        features = features + ds.all_moves[Idx + i].get_features()

      class_last = ds.all_moves[Idx + sequence_classifier.num_time_steps].class_
  
      if class_first == class_last:
        self.alldata.appendLinked(features, [ds.get_classes().index(ds.all_moves[Idx].class_)])
        self.alldata.newSequence()
      

    self.tstdata, self.trndata = self.alldata.splitWithProportion(0.25)
    self.trndata._convertToOneOfMany()
    self.tstdata._convertToOneOfMany()

    self.seq_rnn      = None #buildNetwork(num_inputs, 2, self.trndata.outdim, hiddenclass=LSTMLayer,recurrent=True ,outclass=SoftmaxLayer)
    self.create_network(num_inputs)

    self.trainer  = RPropMinusTrainer(module=self.seq_rnn, dataset=self.trndata)

  def create_network(self, num_inputs):

    self.seq_rnn = RecurrentNetwork()

    in_layer        = LinearLayer(num_inputs)
    hidden_LSTM_   = LSTMLayer(24)
    hidden_layer_0  = LinearLayer(12)
    hidden_layer_1  = SigmoidLayer(12)
    output_layer    = LinearLayer(self.trndata.outdim)

    self.seq_rnn.addInputModule(in_layer)
    self.seq_rnn.addModule(hidden_layer_0)
    self.seq_rnn.addModule(hidden_layer_1)
    self.seq_rnn.addOutputModule(output_layer)
  
    #Now add the connections:
    in_to_LTSM = FullConnection(in_layer      , hidden_LSTM)
    LTSM_to_h0 = FullConnection(hidden_LSTM, hidden_layer_0)

    in_to_h0   = FullConnection(in_layer      , hidden_layer_0)
    h0_to_h1   = FullConnection(hidden_layer_0, hidden_layer_1)
    h1_to_out  = FullConnection(hidden_layer_1, output_layer)

    self.seq_rnn.addConnection(in_to_LSTM)
    self.seq_rnn.addConnection(LSTM_to_h0)

    self.seq_rnn.addConnection(in_to_h0)
    self.seq_rnn.addConnection(h0_to_h1)
    self.seq_rnn.addConnection(h1_to_out)

    self.seq_rnn.sortModules()

  def start_training(self):
    f = open("./results/seq_rnn_perf.txt", "w");
    for i in range(200):
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
    self.seq_rnn.sortModules()
    for Idx in range (len(self.tstdata)):
      out = self.seq_rnn.activate(self.tstdata['input'][Idx])
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
    NetworkWriter.writeToFile(self.seq_rnn,  "./results/" + name + "req_rnn.xml")
