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


#user;  gender;  age;  how_tall_in_meters;  weight;  body_mass_index;    
#x1;    y1;    z1;    x2;    y2;    z2;x3;y3;z3;x4;y4; z4;        class
#sitting_down= filter(lambda ll: ll[-1].find("sittingdown") != -1 , sensors)

class move:
  def __init__(self,idx,l):
    self.name   = l[0].strip()
    self.gender = l[1]
    self.age    = l[2]
    self.height = l[3]
    self.weight = float(l[4].replace(",","."))
    self.BMI    = float(l[5].replace(",","."))

    self.x1 = int(l[6])
    self.y1 = int(l[7])
    self.z1 = int(l[8])
 
    self.x2 = int(l[9])
    self.y2 = int(l[10])
    self.z2 = int(l[11])

    self.x3 = int(l[12])
    self.y3 = int(l[13])
    self.z3 = int(l[14])

    self.x4 = int(l[15])
    self.y4 = int(l[16])
    self.z4 = int(l[17])

    self.class_ = l[18].strip() 

  def get_features(self):
    return [self.x1, self.y1, self.z1, 
            self.x2, self.y2, self.z2, 
            self.x3, self.y3, self.z3, 
            self.x4, self.y4, self.z4, 
            self.age, self.BMI]

  def get_num_features(self):
    return len(self.get_features())

  def get_delta(self):
    return 0

  def __str__(self):
    return "name: " + self.name 

  def __repr__(self):
    return self.__str__()

class data_source:
  all_moves = []

  def __init__(self):
    self.fill_data()
    self.classes = list(set(map( lambda item: item.class_, self.all_moves)))
    self.num_features = self.all_moves[0].get_num_features() 
    
  def get_num_classes(self):
    return len(self.classes)
    
  def get_classes(self):
    return self.classes

  def fill_data(self):
    ff = open("./dataset-har-PUC-Rio-ugulino.csv","rb").readlines()
    first_line = ff.pop(0)
    Idx =0
    for row in ff:
      l = row.strip("\r\n").split(";")
      new_move = move(idx = Idx,l = l)
      Idx += 1
      data_source.all_moves.append(new_move)

class simple_classifier():
  def __init__(self, ds): 
    self.alldata = ClassificationDataSet(ds.num_features, 1, nb_classes=ds.get_num_classes(), class_labels=ds.get_classes())
    #Now add the samples to the data set.
    for sample in ds.all_moves:
      self.alldata.addSample(sample.get_features(), [ds.get_classes().index(sample.class_)])

    self.tstdata, self.trndata = self.alldata.splitWithProportion( 0.25 )
    self.trndata._convertToOneOfMany()
    self.tstdata._convertToOneOfMany()

    #print "Number of training patterns: ", len(self.trndata)
    #print "Input and output dimensions: ", self.trndata.indim, self.trndata.outdim
    #print "First sample (input, target, class):"
    #print self.trndata['input'][0], self.trndata['target'][0], self.trndata['class'][0]
    # 5 hidden layers.  
    self.fnn = buildNetwork(self.trndata.indim, 5, self.trndata.outdim, outclass=SoftmaxLayer)
    self.trainer = BackpropTrainer(self.fnn, dataset=self.trndata, momentum=0.1, verbose=True, weightdecay=0.01)

  def start_training(self):
    #Interactively train the data, see how error bars porgress.
    for i in range(20):
      print "training step: " + str(i)
      self.trainer.trainEpochs(1)
      self.trnresult = percentError(self.trainer.testOnClassData(),
                                    self.trndata['class'])
      self.tstresult = percentError(self.trainer.testOnClassData(
           dataset=self.tstdata), self.tstdata['class'])

      print "epoch: %4d" % self.trainer.totalepochs, \
          "  train error: %5.2f%%" % self.trnresult, \
          "  test error: %5.2f%%" % self.tstresult
      pdb.set_trace()
   
def main():
  src = data_source()
  clf = simple_classifier(src)
  clf.start_training() 

if __name__ == "__main__":
    main()
