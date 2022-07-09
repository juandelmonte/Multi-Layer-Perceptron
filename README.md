# Multi Layer Perceptron (MLP)

This fully connected MLP started as a way of understanding the core principles behind Artificial Neural Netwoks, mainly backpropagation.

It's written using mainly numpy objects. 

For those trying to understand them there's plenty of material explaining what they are and how they are being used, I'm not trying to cover that but if you are one of them I hope at least it gets to be helpful in your journey.

Furthermore, I want to showcase some of its applications. I plan to use this kind of model throughout my professional activity and think this could potentially be helpful by bringing clarity and preventing them from being used as black boxes.

## Machine Learning

In the examples the MLP will be used to solve some machine learning problems.

MLPs are an important component of Machine learning and AI systems and its applications can be explained through mathematical optimization, applied-statistics (and linear algebra).

I like the formal definition of Tom M. Mitchell "A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P if its performance at tasks in T, as measured by P, improves with experience E." (Wikipedia).

You will see how MLP weights and bias change when they are trained with data, the MLP improves with experience, by being more accurate in classification models or simply by producing outputs closer to the real values we want to get in regression models.

Remember that we want to generalize to unknown data, we want the model to be good at predicting from data it did not see or was trained with before. Evaluation might also require some statistical approach.

## Feedforward

To produce the output, a set of transformations are made to the input. This is done in the feedforward method. In this model every node in a layer has the same activation function, then it's simpler to work with numpy matrices and arrays.

## Stochastic approximation of gradient descent optimization

A gradient can be thought of as a vector pointing to where some function grows faster.

A cost function can be thought of as the difference between the MLP output and the output we want it to have.

By moving (small steps) towards the opposite direction of the gradient of a cost function we are making the MLP produce outputs closer to the ones we want it to have. When the MLP is training it's updating its weights (W) and biases (b) to do so. The output of the MLP is actually a function of the input and these W and b, in deep neural networks this is a set of functions that are being applied sequentially or nested functions.

If instead of using the entire dataset random samples are used to train the MLP, then it's known as stochastic. This technique improves performance and can also improve generalization.

## Backward Propagation of Errors

A layer's output will be a function of its W, b and inputs.

Errors can be propagated as the input of a layer is the output of the previous layer.

The gradient is explained with derivatives and the chain rule is applied.