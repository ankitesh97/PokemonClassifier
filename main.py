
import pickle
from NeuralNetwork import NeuralNet
import json

dictpok = json.loads(open('mapclasses.txt','r').read())


def _search_for_key(s_value):
    for key,value in dictpok.iteritem():
        if value == s_value+2:
            return key


def _map_to_pokemon(data):
    pokeclasses = []
    for x in data:
        pokeclasses.append(_search_for_key(x))
    return pokeclasses




def main():
    #load the model from pickle
    data = sio.loadmat('data.mat')
    X = data['X']
    model_str = open('model','r').read()
    model = pickle.loads(model_str)
    predictions = model.predict(X)
    all_predictions = map(_map_to_pokemon,predictions)
    print all_predictions

if __name__ == '__main__':
    main()
