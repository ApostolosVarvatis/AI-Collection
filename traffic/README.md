To begin with all of my tests used the optimizer "adam" which is a popular optimization algorithm that computes adaptive learning rates for each weigh
and i also found to be known for its efficiency and robustness across different types of deep learning models.
The "categorial_crossentropy" loss function is used for multi-class classification problems and for metrics i used accuracy as shown in the lectures.

I alwyas used a dropout of "0.5" to avoid overfitting and regularize data. In addition flattening was always done after the last max-pooling layer.
The convolution layer had a kernel of (3, 3) and the pooling a size of (2, 2).
For all the activation functions except the output layer function i used "ReLU" to set all negative values to zero and keep the positive values unchanged.
At the end, the output layer used the softmax activation function which is optimal for multi-class classification problem and ensures that probabilities sum up to 1.

After thoses fixed variables i tried playing with number of convolutional layers, poooling layers, hidden layers and all of their sizes per layer.

Firstly, i tried one convolutional layer with 43 filters, one max pooling layer and one hidden layer of 43 * 4 units after flattening.
In every epoch the results where the approximately the same around 0.5600 so i understood something was off. 

I immediately added another convolution and max-pooling layer and got normal looking results with the accuracy being 0.9380.

After that i tried adding more filters (64) on both the convolutional layers. The results dropped to 0.9050 so i changed that setting back.

Then i tried a lot more units on the hidden layer (512) which proved even worst with the accuracy being 0.8900.

I then lowered the units to 344 proving to be the most successful test yet with the accuracy being 0.9450.

Furthermore adding one more convolutional and max-pooling layer lower my results so i sticked with 2 each.

So at the end i ended up with an accuracy of 0.9660 a loss of 0.1698 with my best test being on the current code that is submitted.