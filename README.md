# NeuralNetwork_Numbers
Neural Network to recognise human handwritten numbers

This Neural Network has 3 layers (input 784, hidden 200, output 10)
Exemple of number(28*28px): 
![Exemple](https://github.com/DmitriyLeaf/NeuralNetwork_Numbers/blob/master/examlNum.PNG)

It used sigmoid function as an activation function.

There is some results with diffent learning rate and epohs: <br>
-----1-------3-------5-------8       <br>
0.0 [0.9519, 0.9545, 0.9555, 0.9498] <br>
0.1 [0.9532, 0.9525, 0.9452, 0.9533] <br>
0.2 [0.9519, 0.9469, 0.9501, 0.9567] <br>
0.3 [0.957,  0.9462, 0.9501, 0.953]  <br>
0.4 [0.9503, 0.945,  0.951,  0.9552] <br>

Here is testing and training data:
https://pjreddie.com/media/files/mnist_train.csv
https://pjreddie.com/media/files/mnist_test.csv

Packages:
numpy
scipy.special
matplotlib.pyplot
