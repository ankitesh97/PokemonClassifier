
import numpy as np
import pandas as pd
import scipy.io as sio
import json

# classes length 17 i.e 2-19
# ['Ghost', 'Steel','Dark','Electric','Ice','Normal','Fire','Psychic','Poison','Dragon','Water','Fighting', 'Rock',
#  'Fairy','Grass','Bug','Ground']
mapClassesToInt = {}

def _make_classes(data):
    dataGrpBy = data.groupby(['Type 1'])
    classes = dataGrpBy.groups.keys()
    i=2
    for key in classes:
        mapClassesToInt[key] = i
        i += 1
    ClassesMap = open('mapclasses.txt','w')
    ClassesMap.write(json.dumps(mapClassesToInt))
    return mapClassesToInt

def _split(data,classes,ratio=0.7):

    #for basically every class
    dataByClass = data.groupby(['Type 1'])
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
    for key in classes:
        currClassdf = dataByClass.get_group(key)
        currClassdf = currClassdf.sample(frac=1) #shufflw rows
        length = int(ratio*currClassdf.shape[0]) #total entries
        train_df = train_df.append(currClassdf.iloc[:length])
        test_df = test_df.append(currClassdf.iloc[length:])

    train_df = train_df.sample(frac=1)
    test_df = test_df.sample(frac=1)
    return train_df, test_df

def _map_to_class(classType):
    return mapClassesToInt[classType]


def _to_model_format(train,test):
    y_traindf = train['Type 1']
    y_testdf = test['Type 1']
    train.drop('Type 1',axis=1,inplace=True)
    test.drop('Type 1',axis=1,inplace=True)
    X_train = np.array(train)
    X_test = np.array(test)
    y_train = np.array(y_testdf)
    y_test = np.array(y_testdf)
    #save it in a file
    final_train = {"desc":['HP','Attack','Defence','Sp.Atk','Sp.Def','Speed']}
    final_train['X'] = X_train
    final_train['y'] = np.array(map(_map_to_class,y_train))[np.newaxis].T
    final_test = {"desc":['HP','Attack','Defence','Sp.Atk','Sp.Def','Speed']}
    final_test['X'] = X_test
    final_test['y'] =np.array(map(_map_to_class,y_test))[np.newaxis].T
    sio.savemat('train.mat',final_train)
    sio.savemat('test.mat',final_test)

    return (X_train,y_train),(X_test,y_test)

def preprocess():
    #load data
    data_all = pd.read_csv('Pokemon.csv')
    #drop name,type2, generation and is_legendary
    data = data_all.drop(data_all.columns[[1,3,4,11,12]],axis=1)
    #now to remove duplicate entries i.e entries with same id
    data = data.drop_duplicates(data.columns[[0]])
    #set index as the  Pokemon id
    data = data.set_index(['#'])
    #after viewing dataset only got 3 points for flying points so decide to remove one since it had primary type as
    #flying and had no other type i.e tornodus, where as the other two had second type as dragon type i.e noivern
    #noibat....ids 641, 714, 715
    data.loc[[714,715],'Type 1'] = 'Dragon'
    data.drop([641],inplace=True)
    #now to make classes that we are going to predict
    classes = _make_classes(data)
    #type to split data to test and train
    train, test = _split(data,classes,ratio=0.7)
    #save the train and test stuff
    train.to_csv('traindf.csv')
    test.to_csv('testdf.csv')
    #change to numpy array and return
    trainN, testN = _to_model_format(train,test)
    return trainN, testN

if __name__ == '__main__':
    preprocess()
