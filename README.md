# PokemonClassifier
A neural network model ( from scratch) to classify pokemon based on their primary type, i.e(type 1)

## About the data set
The data set was taken from kaggle, it is the pokemon.csv file, it cotained 800 pokemons with their stats  

## Steps
1). After preprocessing/cleaning the data there were around 720 data points  
2). The main task was to classify the pokemons based on their primary types  
3). There are 17 types of pokemon (classify), there were 7 features intially but chose 5 after dimensionality reduction.  
4). Trained a Neural network model of 1 hidden layer  
5). model architecture 5x479x17  
6). Used softmax function for calculating probabilities  
7). The model is pickled and named as model
8). The mat files contains the cleaned data  

## Usage
To view the predictions run the main.py just change the filename of your data  

## Results

-Firt the validation was performed by the letting model predict one class and to cross check it with the  
correctly labeled set it gave an accuracy of around 11.02%  
- Then the validation was performed to let the model predict 5 classes with the highest probability  
this gave and accuracy of 39.4%  
- Lastly the validation was performed to let the model predict 6 classes with the highest probability  
this gave and accuracy of 42.61%  
To summarize the Results  

| Classes Predicted  	| Accuracy (%) 	|  
|--------------------	|--------------	|  
| 1                  	| 11.02        	|  
| 5                  	| 39.4         	|  
| 6                  	| 42.61        	|  

## Conclusion
Finally, the Conclusion that I made from the results  
1). It is hard for neural network to predict the correct results since, the number of data points to train  
the model is very less as compared to features and the classes
2). There seems no relation between the stats and the pokemon's class
3). May the by changing the same problem to the image domain(classifying using image) may provide better results.
