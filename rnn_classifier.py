from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.supervised.trainers import RPropMinusTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.datasets 		 import SequentialDataSet
from pybrain.structure.modules import *

class simple_rnn_classifier():
  def __init__(self, ds): 
    self.alldata = SequentialDataSet(ds.num_features, 1)
    # Now add the samples to the data set.
    idx = 1
    self.alldata.newSequence()
    for sample in ds.all_moves:
      self.alldata.addSample(sample.get_features(), [ds.get_classes().index(sample.class_)])
      idx += 1
      if (idx%6 == 0): 
        self.alldata.newSequence()
     

    self.tstdata, self.trndata = self.alldata.splitWithProportion(0.25)
    #print "Number of training patterns: ", len(self.trndata)
    #print "Input and output dimensions: ", self.trndata.indim, self.trndata.outdim
    #print "First sample (input, target, class):"
    #print self.trndata['input'][0], self.trndata['target'][0], self.trndata['class'][0]
    # 5 hidden layers.  
    self.rnn = buildNetwork(self.trndata.indim,
                            3, 
                            self.trndata.outdim,  
                            hiddenclass=LSTMLayer, 
                            outclass=SigmoidLayer, 
                            recurrent=True)
 
    self.rnn.randomize()
    self.trainer = BackpropTrainer(self.nn, dataset=self.trndata, momentum=0.1, verbose=True, weightdecay=0.01)

      
  def start_training(self):
    f = open("./results/rnn_perf.txt", "w");
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
    self.fnn.sortModules()
    for Idx in range (len(self.tstdata)):
      out = self.fnn.activate(self.tstdata['input'][Idx])
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
    NetworkWriter.writeToFile(self.fnn,  "./results/" + name + "rnn.xml")

