

# Lessons 2

Introduction To Neural Network 

## 1- Introduction 

## 2- Classification Problem 1

In this section, it's explained about classifications problems work by solving questions. 

![student-quiz](/Users/mahamatnouralimai/Desktop/1-2018:2019/pytorch_udacity_scholarship_learning_notes/Notes/student-quiz.png)

## 3- Classification Problem 2 

The answer for quiz problems.

## 4- Linear Boundaries 

wx1 + wx2 + b = 0

Wx + b = 0

w = (w1, w2)

x = (x1, x2)

y= label 0 or 1

y = 

## 5- Highest Dimensions 

## 6- Perceptrons 

## 7- Why Neural Networks?

## 8- Perceptrons as Logical Operators

In this lesson, we'll see one of the many great applications of perceptrons. As logical operators! You'll have the chance to create the perceptrons for the most common of these, the **AND**, **OR**, and **NOT** operators. And then, we'll see what to do about the elusive **XOR** operator. Let's dive in! 

- And Perceptrons 


![and-quiz](/Users/mahamatnouralimai/Desktop/1-2018:2019/pytorch_udacity_scholarship_learning_notes/Notes/and-quiz.png)



```python

import pandas as pd

# TODO: Set weight1, weight2, and bias
weight1 = 1.0
weight2 = 1.0
bias =-2.0


# DON'T CHANGE ANYTHING BELOW
# Inputs and outputs
test_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
correct_outputs = [False, False, False, True]
outputs = []

# Generate and check output
for test_input, correct_output in zip(test_inputs, correct_outputs):
    linear_combination = weight1 * test_input[0] + weight2 * test_input[1] + bias
    output = int(linear_combination >= 0)
    is_correct_string = 'Yes' if output == correct_output else 'No'
    outputs.append([test_input[0], test_input[1], linear_combination, output, is_correct_string])

# Print output
num_wrong = len([output[4] for output in outputs if output[4] == 'No'])
output_frame = pd.DataFrame(outputs, columns=['Input 1', '  Input 2', '  Linear Combination', '  Activation Output', '  Is Correct'])
if not num_wrong:
    print('Nice!  You got it all correct.\n')
else:
    print('You got {} wrong.  Keep trying!\n'.format(num_wrong))
print(output_frame.to_string(index=False))

```



- OR Perceptrons 

![xor](/Users/mahamatnouralimai/Desktop/1-2018:2019/pytorch_udacity_scholarship_learning_notes/Notes/xor.png)



- AND to OR Perceptrons  

  The OR perceptron is very similar to an AND perceptron. In the image below, the OR perceptron has the same line as the AND perceptron, except the line is shifted down. What can you do to the weights and/or bias to achieve this? Use the following AND perceptron to create an OR Perceptron.


![and-to-or](/Users/mahamatnouralimai/Desktop/1-2018:2019/pytorch_udacity_scholarship_learning_notes/Notes/and-to-or.png)

- XOR Perceptrons 

![xor](/Users/mahamatnouralimai/Desktop/1-2018:2019/pytorch_udacity_scholarship_learning_notes/Notes/xor.png)

- XOR

  ## Quiz Build an XOR Multi-Layer Perceptrons 

  Now, let's build a multi-layer perceptron from the AND, NOT, and OR perceptrons to create XOR logic!

  The neural network below contains 3 perceptrons, A, B, and C. The last one (AND) has been given for you. The input to the neural network is from the first node. The output comes out of the last node.

  The multi-layer perceptron below calculates XOR. Each perceptron is a logic operation of AND, OR, and NOT. However, the perceptrons A, B, and C don't indicate their operation. In the following quiz, set the correct operations for the perceptrons to calculate XOR.


![xor-quiz](/Users/mahamatnouralimai/Desktop/1-2018:2019/pytorch_udacity_scholarship_learning_notes/Notes/xor-quiz.png)

![Screen Shot 2018-11-10 at 22.59.20](/Users/mahamatnouralimai/Desktop/1-2018:2019/pytorch_udacity_scholarship_learning_notes/Notes/Screen Shot 2018-11-10 at 22.59.20.png)

- NOT Perceptrons 

  ```python
  import pandas as pd
  
  # TODO: Set weight1, weight2, and bias
  weight1 = 0.0
  weight2 = 0.0
  bias = 0.0
  
  
  # DON'T CHANGE ANYTHING BELOW
  # Inputs and outputs
  test_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
  correct_outputs = [True, False, True, False]
  outputs = []
  
  # Generate and check output
  for test_input, correct_output in zip(test_inputs, correct_outputs):
      linear_combination = weight1 * test_input[0] + weight2 * test_input[1] + bias
      output = int(linear_combination >= 0)
      is_correct_string = 'Yes' if output == correct_output else 'No'
      outputs.append([test_input[0], test_input[1], linear_combination, output, is_correct_string])
  
  # Print output
  num_wrong = len([output[4] for output in outputs if output[4] == 'No'])
  output_frame = pd.DataFrame(outputs, columns=['Input 1', '  Input 2', '  Linear Combination', '  Activation Output', '  Is Correct'])
  if not num_wrong:
      print('Nice!  You got it all correct.\n')
  else:
      print('You got {} wrong.  Keep trying!\n'.format(num_wrong))
  print(output_frame.to_string(index=False))
  ```


## 9- Perceptron Trick 

In the last section you used your logic and your mathematical knowledge to create perceptrons for some of the most common logical operators. In real life, though, we can't be building these perceptrons ourselves. The idea is that we give them the result, and they build themselves. For this, here's a pretty neat trick that will help us.

![perceptronquiz](/Users/mahamatnouralimai/Desktop/1-2018:2019/pytorch_udacity_scholarship_learning_notes/Notes/perceptronquiz.png)



### Time for some math!

Now that we've learned that the points that are misclassified, want the line to move closer to them, let's do some math. The following video shows a mathematical trick that modifies the equation of the line, so that it comes closer to a particular point.



For the second example, where the line is described by 3x1+ 4x2 - 10 = 0, if the learning rate was set to 0.1, how many times would you have to apply the perceptron trick to move the line to a position where the blue point, at (1, 1), is correctly classified?

## 10- Perceptrons Algorithm

And now, with the perceptron trick in our hands, we can fully develop the perceptron algorithm! The following video will show you the pseudocode, and in the quiz below, you'll have the chance to code it in Python.

Time to code! In this quiz, you'll have the chance to implement the perceptron algorithm to separate the following data (given in the file data.csv).

##  ![points](/Users/mahamatnouralimai/Desktop/1-2018:2019/pytorch_udacity_scholarship_learning_notes/Notes/points.png)



Recall that the perceptron step works as follows. For a point with coordinates (p,q)(p,q), label yy, and prediction given by the equation y^=step(w1x1+w2x2+b)y^=step(w1x1+w2x2+b):

- If the point is correctly classified, do nothing.
- If the point is classified positive, but it has a negative label, subtract αp,αq,αp,αq, and αα from w1,w2,w1,w2,and bb respectively.
- If the point is classified negative, but it has a positive label, add αp,αq,αp,αq, and αα to w1,w2,w1,w2, and bbrespectively.

Then click on `test run` to graph the solution that the perceptron algorithm gives you. It'll actually draw a set of dotted lines, that show how the algorithm approaches to the best solution, given by the black solid line.

Feel free to play with the parameters of the algorithm (number of epochs, learning rate, and even the randomizing of the initial parameters) to see how your initial conditions can affect the solution!



```python
import numpy as np
# Setting the random seed, feel free to change it and see different solutions.
np.random.seed(42)

def stepFunction(t):
    if t >= 0:
        return 1
    return 0

def prediction(X, W, b):
    return stepFunction((np.matmul(X,W)+b)[0])

# TODO: Fill in the code below to implement the perceptron trick.
# The function should receive as inputs the data X, the labels y,
# the weights W (as an array), and the bias b,
# update the weights and bias W, b, according to the perceptron algorithm,
# and return W and b.
def perceptronStep(X, y, W, b, learn_rate = 0.01):
    # Fill in code
    def perceptronStep(X, y, W, b, learn_rate = 0.01):
    for i in range(len(X)):
        y_hat = prediction(X[i],W,b)
        if y[i]-y_hat == 1:
            W[0] += X[i][0]*learn_rate
            W[1] += X[i][1]*learn_rate
            b += learn_rate
        elif y[i]-y_hat == -1:
            W[0] -= X[i][0]*learn_rate
            W[1] -= X[i][1]*learn_rate
            b -= learn_rate
    return W, b
    
# This function runs the perceptron algorithm repeatedly on the dataset,
# and returns a few of the boundary lines obtained in the iterations,
# for plotting purposes.
# Feel free to play with the learning rate and the num_epochs,
# and see your results plotted below.
def trainPerceptronAlgorithm(X, y, learn_rate = 0.01, num_epochs = 25):
    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])
    W = np.array(np.random.rand(2,1))
    b = np.random.rand(1)[0] + x_max
    # These are the solution lines that get plotted below.
    boundary_lines = []
    for i in range(num_epochs):
        # In each epoch, we apply the perceptron step.
        W, b = perceptronStep(X, y, W, b, learn_rate)
        boundary_lines.append((-W[0]/W[1], -b/W[1]))
    return boundary_lines

```



## 11- Non-Linear Regions 

## 12- Error Functions 

## 13- Log-loss Error Functions 

## 14- Discrete vs Continous 

the answer for the lessons 2.14

![exp01](/Users/mahamatnouralimai/Desktop/1-2018:2019/pytorch_udacity_scholarship_learning_notes/Notes/exp01.PNG)

## 15- Softmax 

```python
#Formula of softmax function

import numpy as np

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    expL = np.exp(L)
    sumExpL = sum(expL)
    result = []
    for i in expL:
        result.append(i*1.0/sumExpL)
    return result
    
```



## 16- One- Hot Encoding 



## 17- Maximum Likelihood

# Maximum Likelihood

Probability will be one of our best friends as we go through Deep Learning. In this lesson, we'll see how we can use probability to evaluate (and improve!) our models.

## 18- Maximizing Probablities 

# Maximizing Probabilities

In this lesson and quiz, we will learn how to maximize a probability, using some math. Nothing more than high school math, so get ready for a trip down memory lane!

*Correction:* At 2:18, the top right point should be labelled `-log(0.7)` instead of `-log(0.2)`.

## 19- Cross- Entropy 1

a good model will give low cross entropy. 

a bad model will give us high cross entropy 

goal is to minimizing the cross entropy 

## 20- Cross- Entropy 2

# Cross-Entropy

So we're getting somewhere, there's definitely a connection between probabilities and error functions, and it's called **Cross-Entropy**. This concept is tremendously popular in many fields, including Machine Learning. Let's dive more into the formula, and actually code it!



```python
import numpy as np
# the formula for cross-entropy 
# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy(Y, P):
    Y = np.float_(Y)
    P = np.float_(P)
    return -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))
    
```





## 21- Multi-Class Cross Entropy 



## 22- Logistic Regression  

# Logistic Regression

Now, we're finally ready for one of the most popular and useful algorithms in Machine Learning, and the building block of all that constitutes Deep Learning. The **Logistic Regression** Algorithm. And it basically goes like this:

- Take your data
- Pick a random model
- Calculate the error
- Minimize the error, and obtain a better model
- Enjoy!

### Calculating the Error Function

Let's dive into the details. The next video will show you how to calculate an error function

## 23- Gradient Descent 

In this lesson, we'll learn the principles and the math behind the gradient descent algorithm.

# Gradient Calculation

In the last few videos, we learned that in order to minimize the error function, we need to take some derivatives. So let's get our hands dirty and actually compute the derivative of the error function. The first thing to notice is that the sigmoid function has a really nice derivative. Namely,

\sigma'(x) = \sigma(x) (1-\sigma(x))σ′(x)=σ(x)(1−σ(x))

The reason for this is the following, we can calculate it using the quotient formula:





![img](https://d17h27t6h515a5.cloudfront.net/topher/2017/May/5910e6c6_codecogseqn-49/codecogseqn-49.gif)





And now, let's recall that if we have mm points labelled x^{(1)}, x^{(2)}, \ldots, x^{(m)},x(1),x(2),…,x(m), the error formula is:

E = -\frac{1}{m} \sum_{i=1}^m \left( y_i \ln(\hat{y_i}) + (1-y_i) \ln (1-\hat{y_i}) \right)E=−m1∑i=1m(yiln(yi^)+(1−yi)ln(1−yi^))

where the prediction is given by \hat{y_i} = \sigma(Wx^{(i)} + b).yi^=σ(Wx(i)+b).

Our goal is to calculate the gradient of E,E, at a point x = (x_1, \ldots, x_n),x=(x1,…,xn), given by the partial derivatives

\nabla E =\left(\frac{\partial}{\partial w_1}E, \cdots, \frac{\partial}{\partial w_n}E, \frac{\partial}{\partial b}E \right)∇E=(∂w1∂E,⋯,∂wn∂E,∂b∂E)

To simplify our calculations, we'll actually think of the error that each point produces, and calculate the derivative of this error. The total error, then, is the average of the errors at all the points. The error produced by each point is, simply,

E = - y \ln(\hat{y}) - (1-y) \ln (1-\hat{y})E=−yln(y^)−(1−y)ln(1−y^)

In order to calculate the derivative of this error with respect to the weights, we'll first calculate \frac{\partial}{\partial w_j} \hat{y}.∂wj∂y^. Recall that \hat{y} = \sigma(Wx+b),y^=σ(Wx+b), so:





![img](https://d17h27t6h515a5.cloudfront.net/topher/2017/May/590eac24_codecogseqn-43/codecogseqn-43.gif)





The last equality is because the only term in the sum which is not a constant with respect to w_jwj is precisely w_j x_j,wjxj, which clearly has derivative x_j.xj.

Now, we can go ahead and calculate the derivative of the error EE at a point x,x, with respect to the weight w_j.wj.





![img](https://s3.amazonaws.com/video.udacity-data.com/topher/2018/January/5a716f3e_codecogseqn-60-2/codecogseqn-60-2.png)





A similar calculation will show us that





![img](https://d17h27t6h515a5.cloudfront.net/topher/2017/September/59b75d1d_codecogseqn-58/codecogseqn-58.gif)





This actually tells us something very important. For a point with coordinates (x_1, \ldots, x_n),(x1,…,xn), label y,y, and prediction \hat{y},y^, the gradient of the error function at that point is \left(-(y - \hat{y})x_1, \cdots, -(y - \hat{y})x_n, -(y - \hat{y}) \right).(−(y−y^)x1,⋯,−(y−y^)xn,−(y−y^)). In summary, the gradient is

\nabla E = -(y - \hat{y}) (x_1, \ldots, x_n, 1).∇E=−(y−y^)(x1,…,xn,1).

If you think about it, this is fascinating. The gradient is actually a scalar times the coordinates of the point! And what is the scalar? Nothing less than a multiple of the difference between the label and the prediction. What significance does this have?

# Gradient Descent Step

Therefore, since the gradient descent step simply consists in subtracting a multiple of the gradient of the error function at every point, then this updates the weights in the following way:

w_i' \leftarrow w_i -\alpha [-(y - \hat{y}) x_i],wi′←wi−α[−(y−y^)xi],

which is equivalent to

w_i' \leftarrow w_i + \alpha (y - \hat{y}) x_i.wi′←wi+α(y−y^)xi.

Similarly, it updates the bias in the following way:

b' \leftarrow b + \alpha (y - \hat{y}),b′←b+α(y−y^),

*Note:* Since we've taken the average of the errors, the term we are adding should be \frac{1}{m} \cdot \alpham1⋅α instead of \alpha,α, but as \alphaα is a constant, then in order to simplify calculations, we'll just take \frac{1}{m} \cdot \alpham1⋅α to be our learning rate, and abuse the notation by just calling it \alpha.α.

## 24- Logistic Regression Algorithm 

## 25- Pre -Notebook: Gradient Descent 

## 31 Neural Network Architecture

Ok, so we're ready to put these building blocks together, and build great Neural Networks! (Or Multi-Layer Perceptrons, however you prefer to call them.)

This first two videos will show us how to combine two perceptrons into a third, more complicated one.

here how to create linear model, we are going to combine two linear model into one model or combine regions 

We will use sigmoid function to combine two linear model. 



### Multiple layers

Now, not all neural networks look like the one above. They can be way more complicated! In particular, we can do the following things:

- Add more nodes to the input, hidden, and output layers.
- Add more layers.

### Multi-Class Classification

And here we elaborate a bit more into what can be done if our neural network needs to model data with more than one output.

## 32 FeedForward 

Feedforward is the process neural networks use to turn the input into an output. Let's study it more carefully, before we dive into how to train the networks.

# Error Function

Just as before, neural networks will produce an error function, which at the end, is what we'll be minimizing. The following video shows the error function for a neural network.

## 32 Backprogation 

Now, we're ready to get our hands into training a neural network. For this, we'll use the method known as **backpropagation**. In a nutshell, backpropagation will consist of:

- Doing a feedforward operation.
- Comparing the output of the model with the desired output.
- Calculating the error.
- Running the feedforward operation backwards (backpropagation) to spread the error to each of the weights.
- Use this to update the weights, and get a better model.
- Continue this until we have a model that is good.

Sounds more complicated than what it actually is. Let's take a look in the next few videos. The first video will show us a conceptual interpretation of what backpropagation is.

