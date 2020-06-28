# Name
Farmer Exploitation Prevention ML model

# Description
The project is based on a regression model of machine learning which uses the crop name, state name and values of cost of cultivation and cost of production
to predict the yield of the crop. With the prediction of yield of crop available the farmers can now tally their profits or losses and accordingly decide if
the given crop is suitable for cultivation.

# Software Required
Any python IDE(e.g. Spyder/PyCharm)

# Working
Since the dataset used for building the model contains only 50 entries the use of neural network to get the output was avoided. The model just uses simple
regression models to obtain the output therefore in order to decide which model suits the best I have implemented every regression model I know of and then
calculated the mean square error as well as accuracies for each of them.Among these the best accuracies and the least mean square error was observed in the
random forest regression model. Hence the random forest model is best suited for the above dataset.
Since there are multiple independant variables the graph representation was not possible.

# Progress
The machine learning model is ready.
I am learning front end in order to apply the ml model for usage.
