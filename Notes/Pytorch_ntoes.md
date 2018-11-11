

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

## 11- Non-Linear Regions 

## 12- Error Functions 

## 13- Log-loss Error Functions 

## 14- Discrete vs Continous 

## 15- Softmax 

## 16- One- Hot Encoding 

## 17- Maximum Likelihood

