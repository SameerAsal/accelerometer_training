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
            self.x4, self.y4, self.z4] 
#            self.age, self.BMI]

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

      if not (l[0].strip()  == "debora"):
        continue
#      if (l[18].strip() == "walking") or (l[18].strip() == "sitting"):
#        print "ignoring walking\sitting  state"
#        continue

      new_move = move(idx = Idx,l = l)
      Idx += 1
      data_source.all_moves.append(new_move)
    print "Number of elements in data source", len(data_source.all_moves)

