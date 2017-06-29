# Getting Started With TensorFlow（TensorFlow入门）

This guide gets you started programming in TensorFlow. Before using this guide,[install TensorFlow](https://www.tensorflow.org/install/index). To get the most out of this guide, you should know the following:

本教程指导你如何用TensorFlow开始编程。在开始前，请先安装TensorFlow。为了更好的使用本教程，你还应该知道下面的知识：

* How to program in Python.
  如何使用Python编程
* At least a little bit about arrays.

  了解一些数组知识
* Ideally, something about machine learning. However, if you know little or nothing about machine learning, then this is still the first guide you should read.

  如果还知道一些机器学习的知识就更好了。当然，如果你从未了解机器学习，本教程也是你应该阅读的第一个教程。

TensorFlow provides multiple APIs. The lowest level API--TensorFlow Core-- provides you with complete programming control. We recommend TensorFlow Core for machine learning researchers and others who require fine levels of control over their models. The higher level APIs are built on top of TensorFlow Core. These higher level APIs are typically easier to learn and use than TensorFlow Core. In addition, the higher level APIs make repetitive tasks easier and more consistent between different users. A high-level API like tf.contrib.learn helps you manage data sets, estimators, training and inference. Note that a few of the high-level TensorFlow APIs--those whose method names contain`contrib`-- are still in development. It is possible that some`contrib`methods will change or become obsolete in subsequent TensorFlow releases.

TensorFlow提供了多种API接口。最底层的API--TensorFlow Core-- 提供了完整的编程控制。我们推荐机器学习的研究人员和那些需要更好地控制他们的模型的人使用TensorFlow Core。有些更高级别的API比TensorFlow Core更容易学习和使用，而且，这些更高级别的API使不同的用户间的重复任务更容易实现及保持一致性。一个高层API如tf.contrib.learn将帮助你管理数据集，估算，训练和推论。值得注意的是有少数几个高层TensorFlow API的名字包含了contrib --表示还在开发中。有可能这些contrib方法会在以后的TensorFlow版本中被修改或废除。

This guide begins with a tutorial on TensorFlow Core. Later, we demonstrate how to implement the same model in tf.contrib.learn. Knowing TensorFlow Core principles will give you a great mental model of how things are working internally when you use the more compact higher level API.

本教程从TensorFlow Core的入门指导开始，然后我们将演示如何用tf.contrib.learn去实现同样的模型。知道了TensorFlow Core的原理将会在你脑子里建立一个模型，这个模型会帮助你理解当你使用封装更好的高层API时，它们是如何工作的。

# Tensors {#tensors}（张量 ）

The central unit of data in TensorFlow is the **tensor**. A tensor consists of a set of primitive values shaped into an array of any number of dimensions. A tensor's **rank **is its number of dimensions. Here are some examples of tensors:

TensorFlow的核心数据是张量。张量是用形状为任意维度数组表示的一组基本数据。张量的阶就是形状的维度。下面是一些张量的例子：

```Python
3 # a rank 0 tensor; this is a scalar with shape []
3 # 0阶张量，一个标量，形状为[]

[1. ,2., 3.] # a rank 1 tensor; this is a vector with shape [3]
[1. ,2., 3.] # 1阶张量，一个矢量，形状为[3]

[[1., 2., 3.], [4., 5., 6.]] # a rank 2 tensor; a matrix with shape [2, 3]
[[1., 2., 3.], [4., 5., 6.]] # 2阶张量；一个矩阵，形状为[2, 3]

[[[1., 2., 3.]], [[7., 8., 9.]]] # a rank 3 tensor with shape [2, 1, 3]
[[[1., 2., 3.]], [[7., 8., 9.]]] # 一个3阶张量，形状为[2, 1, 3]
```

## TensorFlow Core tutorial {#tensorflow_core_tutorial}(TensorFlow Core教程 )

### Importing TensorFlow {#importing_tensorflow}(引入TensorFlow )

The canonical import statement for TensorFlow programs is as follows:

典型的引入TensorFlow的语句是：

```python
import tensorflow as tf
```

This gives Python access to all of TensorFlow's classes, methods, and symbols. Most of the documentation assumes you have already done this.

以上语句让Python能访问所有TensorFlow类，方法和符号。所有的文档都假设你已经做了这一步。

### The Computational Graph {#the_computational_graph}(计算图 )

You might think of TensorFlow Core programs as consisting of two discrete sections:  
你可以认为TensorFlow Core程序由两个独立部分组成：

1. Building the computational graph.

   建立计算图

2. Running the computational graph.

   运行计算图.

A **computational graph **is a series of TensorFlow operations arranged into a graph of nodes. Let's build a simple computational graph. Each node takes zero or more tensors as inputs and produces a tensor as an output. One type of node is a constant. Like all TensorFlow constants, it takes no inputs, and it outputs a value it stores internally. We can create two floating point Tensors node1 and node2 as follows:

计算图是布置在节点图中的一系列TensorFlow操作。 我们来构建一个简单的计算图。 每个节点采用零个或多个张量作为输入，并产生张量作为输出。 一种类型的节点是一个常数。 像所有TensorFlow常量一样，它不需要任何输入，它输出一个内部存储的值。 我们可以创建两个浮点张量node1和node2，如下所示：

```python
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
print(node1, node2)
```

The final print statement produces
最终打印语句输出

```
Tensor("Const:0", shape=(), dtype=float32) Tensor("Const_1:0", shape=(), dtype=float32)
```

Notice that printing the nodes does not output the values `3.0` and `4.0` as you might expect. Instead, they are nodes that, when evaluated, would produce 3.0 and 4.0, respectively. To actually evaluate the nodes, we must run the computational graph within a **session**. A session encapsulates the control and state of the TensorFlow runtime.
请注意，打印节点不会按你想象的那样输出值是3.0和4.0。 相反，它们是节点，只有在求值时才分别产生3.0和4.0。 要实际对节点的值，我们必须在会话中运行计算图。 会话封装了TensorFlow运行时的控制和状态。

The following code creates a `Session` object and then invokes its `run` method to run enough of the computational graph to evaluate `node1` and `node2`. By running the computational graph in a session as follows:

以下代码创建一个Session对象，然后调用其run方法运行足够的计算图来计算node1和node2。 通过如下方式在会话中运行计算图：

```python
sess = tf.Session()
print(sess.run([node1, node2]))
```

we see the expected values of 3.0 and 4.0:

我们看到了期待的值3.0和4.0

```python
[3.0, 4.0]
```

We can build more complicated computations by combining `Tensor` nodes with operations (Operations are also nodes.). For example, we can add our two constant nodes and produce a new graph as follows:

我们可以通过组合操作张量的节点构造更为复杂的计算（操作也是节点）。 例如，我们对两个常量节点相加并生成一个新的图，如下所示：

```python
node3 = tf.add(node1, node2)
print("node3: ", node3)
print("sess.run(node3): ",sess.run(node3))
```

The last two print statements produce

 最后两个print语句输出node3：

```python
node3:  Tensor("Add:0", shape=(), dtype=float32)
sess.run(node3):  7.0
```

TensorFlow provides a utility called TensorBoard that can display a picture of the computational graph. Here is a screenshot showing how TensorBoard visualizes the graph:

TensorFlow提供了一个名为TensorBoard的实用程序，可以显示计算图的图片。这里有一个屏幕截图，它显示了TensorBoard如何可视化计算图的：

![TensorBoard screenshot](https://www.tensorflow.org/images/getting_started_add.png)

As it stands, this graph is not especially interesting because it always produces a constant result. A graph can be parameterized to accept external inputs, known as **placeholders**. A **placeholder** is a promise to provide a value later.

就像这样，这个图并不是特别有趣，因为它总是产生一个恒定的结果。 计算图可以参数化为接受外部输入，称为占位符(placeholder)。 占位符是一个声明，约定其值将在以后提供。

```python
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # + provides a shortcut for tf.add(a, b)
```

The preceding three lines are a bit like a function or a lambda in which we define two input parameters (a and b) and then an operation on them. We can evaluate this graph with multiple inputs by using the feed_dict parameter to specify Tensors that provide concrete values to these placeholders:

前面的三行有点像一个函数或一个lambda表达式，其中我们定义了两个输入参数（a和b），然后对它们进行一个操作。 我们可以通过使用feed_dict参数来指定为这些占位符提供具体值的张量来计算图表：

```python
print(sess.run(adder_node, {a: 3, b:4.5}))
print(sess.run(adder_node, {a: [1,3], b: [2, 4]}))
```

resulting in the output

输出结果

```python
7.5
[ 3.  7.]
```

In TensorBoard, the graph looks like this:
在TensorBoard中可看到如下的图:

![TensorBoard screenshot](https://www.tensorflow.org/images/getting_started_adder.png)

We can make the computational graph more complex by adding another operation. For example,

我们可以通过增加其他操作来构建更复杂的计算图。比如：

```python
add_and_triple = adder_node * 3.
print(sess.run(add_and_triple, {a: 3, b:4.5}))
```

produces the output

产生输出

```python
22.5
```

The preceding computational graph would look as follows in TensorBoard:

处理的计算图在TensorBoard中表示如下：

![TensorBoard screenshot](https://www.tensorflow.org/images/getting_started_triple.png)

In machine learning we will typically want a model that can take arbitrary inputs, such as the one above. To make the model trainable, we need to be able to modify the graph to get new outputs with the same input. **Variables** allow us to add trainable parameters to a graph. They are constructed with a type and initial value:

在机器学习中，我们通常会想要一个可以接受任意输入的模型，比如上面的一个。 为了使模型可训练，我们需要修改图使得相同的输入以获得新的输出。 变量（Variables)允许我们向图添加可训练的参数。 它们由类型和初始值来构造：

```python
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b
```

Constants are initialized when you call `tf.constant`, and their value can never change. By contrast, variables are not initialized when you call `tf.Variable`. To initialize all the variables in a TensorFlow program, you must explicitly call a special operation as follows:

当调用tf.constant时，常量被初始化，它们的值永远不会改变。 相比之下，当您调用tf.Variable时，变量不会被初始化。 要初始化TensorFlow程序中的所有变量，必须显式调用特殊操作，如下所示：

```python
init = tf.global_variables_initializer()
sess.run(init)
```

It is important to realize `init` is a handle to the TensorFlow sub-graph that initializes all the global variables. Until we call `sess.run`, the variables are uninitialized.

Since `x` is a placeholder, we can evaluate `linear_model` for several values of `x` simultaneously as follows:

实现init是非常重要的，它是TensorFlow子图初始化所有的全局变量的一个句柄。 因为在调用sess.run之前，这些变量是未初始化的。

由于x是占位符，我们可以同时求出linear_model线性模型的多个x值，如下所示：

```python
print(sess.run(linear_model, {x:[1,2,3,4]}))
```

to produce the output

将产生输出

```python
[ 0.          0.30000001  0.60000002  0.90000004]
```

We've created a model, but we don't know how good it is yet. To evaluate the model on training data, we need a `y`placeholder to provide the desired values, and we need to write a loss function.

A loss function measures how far apart the current model is from the provided data. We'll use a standard loss model for linear regression, which sums the squares of the deltas between the current model and the provided data. `linear_model - y` creates a vector where each element is the corresponding example's error delta. We call `tf.square` to square that error. Then, we sum all the squared errors to create a single scalar that abstracts the error of all examples using `tf.reduce_sum`:

我们创建了一个模型，但是我们不知道它有多好。 为了评估培训数据模型，我们需要一个y占位符来提供期望值，另外我们还需要编写一个损失函数。

损失函数测量当前模型与提供的数据之间的距离。 我们将使用线性回归的标准损耗模型，它对当前模型和提供的数据之间的差的平方求和。 linear_model - y创建一个向量，其中每个元素都是对应的示例的错误增量。 我们调用tf.square计算平方误差。 然后，我们使用tf.reduce_sum对所有示例的平方误差求和，于是得到一个单一的标量：

```python
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))
```

producing the loss value

输出损失数值

```python
23.66
```

We could improve this manually by reassigning the values of `W` and `b` to the perfect values of -1 and 1. A variable is initialized to the value provided to `tf.Variable` but can be changed using operations like `tf.assign`. For example,`W=-1` and `b=1` are the optimal parameters for our model. We can change `W` and `b` accordingly:

我们可以通过手动方式将W和b的值重新赋上完美的-1和1的值来改进。变量通过tf.Variable提供初始值，但它可以通过像tf.assign这样的操作进行更改。 例如，W = -1和b = 1是我们模型的最优参数。 于是，我们修改W和b：

```python
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))
```

The final print shows the loss now is zero.

最后打印显示错误率为0

```
0.0

```

We guessed the "perfect" values of `W` and `b`, but the whole point of machine learning is to find the correct model parameters automatically. We will show how to accomplish this in the next section.

我们猜测到W和B的“完美”值，但机器学习的重点是自动找到正确的模型参数。 我们将在下一节中展示如何完成这一点。

## tf.train API

A complete discussion of machine learning is out of the scope of this tutorial. However, TensorFlow provides **optimizers**that slowly change each variable in  order to minimize the loss function. The simplest optimizer is **gradient descent**. It modifies each variable according to the magnitude of the derivative of loss with respect to that variable. In general, computing symbolic derivatives manually is tedious and error-prone. Consequently, TensorFlow can automatically produce derivatives given only a description of the model using the function `tf.gradients`. For simplicity, optimizers typically do this for you. For example,

机器学习的完整讨论超出了本教程的范围。 然而，TensorFlow提供了优化器，可以缓慢地更改每个变量，以便最小化损失函数。 最简单的优化器是梯度下降。 它根据相对于该变量的损失导数的大小修改每个变量。 通常，手动计算符号导数是冗长乏味且容易出错的。 因此，TensorFlow可以使用函数tf.gradients自动生成仅给出模型描述的导数。 为了简单起见，优化器通常为您做这个。 例如，

```python
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
```

```python
sess.run(init) # reset values to incorrect defaults.
for i in range(1000):
  sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})

print(sess.run([W, b]))
```

results in the final model parameters:

结果是最终的模型参数：

```python
[array([-0.9999969], dtype=float32), array([ 0.99999082],
 dtype=float32)]
```

Now we have done actual machine learning! Although doing this simple linear regression doesn't require much TensorFlow core code, more complicated models and methods to feed data into your model necessitate more code. Thus TensorFlow provides higher level abstractions for common patterns, structures, and functionality. We will learn how to use some of these abstractions in the next section.

现在我们已经做了实际的机器学习！ 尽管这样做简单的线性回归并不需要太多的TensorFlow核心代码，但更复杂的模型和方法将数据输入到模型中需要更多的代码。 因此，TensorFlow为常见的模式，结构和功能提供了更高级别的抽象。 我们将在下一节中学习如何使用这些抽象中的一些。

### Complete program（完整的代码）

The completed trainable linear regression model is shown here:

完整的可训练线性模型如下：

```python
import numpy as np
import tensorflow as tf

# Model parameters
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)
# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
# training data
x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]
# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
for i in range(1000):
  sess.run(train, {x:x_train, y:y_train})

# evaluate training accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x:x_train, y:y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
```

When run, it produces

运行时输出：

```python
W: [-0.9999969] b: [ 0.99999082] loss: 5.69997e-11
```

Notice that the loss is a very small number (close to zero). If you run this program your loss will not be exactly the same, because the model is initialized with random values.

请注意，损失是非常小的数字（接近零）。 如果运行此程序，损失将不会完全相同，因为模型是用随机值初始化的

This more complicated program can still be visualized in TensorBoard 

这个更复杂的程序仍然可以使用TensorBoard可视化



![TensorBoard final model visualization](https://www.tensorflow.org/images/getting_started_final.png)

## `tf.contrib.learn`

`tf.contrib.learn` is a high-level TensorFlow library that simplifies the mechanics of machine learning, including the following:

tf.contrib.learn是一个高级TensorFlow库，它简化了机器学习的机制，其中包括：

- running training loops

  执行训练循环

- running evaluation loops

  执行求值循环

- managing data sets

  管理数据集

- managing feeding

  管理供给



tf.contrib.learn defines many common models.

tf.contrib.learn定义了许多常见的模型。

### Basic usage（基本用法）

Notice how much simpler the linear regression program becomes with `tf.contrib.learn`:

让我们看看使用tf.contrib.learn能多大程度上简化线性回归程序：

```python
import tensorflow as tf
# NumPy is often used to load, manipulate and preprocess data.
import numpy as np

# Declare list of features. We only have one real-valued feature. There are many
# other types of columns that are more complicated and useful.
# 声明特性列表，我们只有一个实值特性。 还有许多其他更为复杂和有用的特性列类型。
features = [tf.contrib.layers.real_valued_column("x", dimension=1)]

# An estimator is the front end to invoke training (fitting) and evaluation
# (inference). There are many predefined types like linear regression,
# logistic regression, linear classification, logistic classification, and
# many neural network classifiers and regressors. The following code
# provides an estimator that does linear regression.
# 评价器是调用训练（拟合）和评估（推理）的前端
# 有许多预定义类型，如线性回归，逻辑回归，线性分类，逻辑分类和
# 许多神经网络分类器和回归器。 以下代码提供一个进行线性回归的评价器。
estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)

# TensorFlow provides many helper methods to read and set up data sets.
# Here we use two data sets: one for training and one for evaluation
# We have to tell the function how many batches
# of data (num_epochs) we want and how big each batch should be.
# TensorFlow提供了许多帮助方法来读取和设置数据集。
# 这里我们使用两个数据集：一个用于训练，一个用于评估
# 我们必须告诉函数我们想要多少批次的数据（num_epochs），每个批次应该有多大。
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])
input_fn = tf.contrib.learn.io.numpy_input_fn({"x":x_train}, y_train,
                                              batch_size=4,
                                              num_epochs=1000)
eval_input_fn = tf.contrib.learn.io.numpy_input_fn(
    {"x":x_eval}, y_eval, batch_size=4, num_epochs=1000)

# We can invoke 1000 training steps by invoking the  method and passing the
# training data set.
# 我们将训练数据传给训练方法，并调用它训练1000次
estimator.fit(input_fn=input_fn, steps=1000)

# Here we evaluate how well our model did.
# 这里我们估算我们的模型有多好
train_loss = estimator.evaluate(input_fn=input_fn)
eval_loss = estimator.evaluate(input_fn=eval_input_fn)
print("train loss: %r"% train_loss)
print("eval loss: %r"% eval_loss)
```

When run, it produces

运行，输出

```python
    train loss: {'global_step': 1000, 'loss': 4.3049088e-08}
    eval loss: {'global_step': 1000, 'loss': 0.0025487561}
```

Notice how our eval data has a higher loss, but it is still close to zero. That means we are learning properly.

注意到我们的评估数据有比较高的损失，但是它还是趋近于0的。也就是说我们的学习是正确的。

### A custom model

`tf.contrib.learn` does not lock you into its predefined models. Suppose we wanted to create a custom model that is not built into TensorFlow. We can still retain the high level abstraction of data set, feeding, training, etc. of`tf.contrib.learn`. For illustration, we will show how to implement our own equivalent model to `LinearRegressor`using our knowledge of the lower level TensorFlow API.

tf.contrib.learn不会将你限制在其预定义的模型中。 假设我们想创建一个没有内置到TensorFlow中的自定义模型。 我们仍然可以保留tf.contrib.learn的数据集，供给，训练等的高级抽象。 为了说明，我们将展示如何使用较低级别TensorFlow API的知识，实现LinearRegressor等效的自定义模型。

To define a custom model that works with `tf.contrib.learn`, we need to use `tf.contrib.learn.Estimator`. `tf.contrib.learn.LinearRegressor` is actually a sub-class of `tf.contrib.learn.Estimator`. Instead of sub-classing `Estimator`, we simply provide `Estimator` a function `model_fn` that tells `tf.contrib.learn` how it can evaluate predictions, training steps, and loss. The code is as follows:

要定义使用tf.contrib.learn的自定义模型，我们需要使用tf.contrib.learn.Estimator。 tf.contrib.learn.LinearRegressor实际上是一个tf.contrib.learn.Estimator的子类。我们不是通过Estimator的子类，而是简单地给Estimator提供一个函数model_fn来告诉tf.contrib.learn如何评估预测，训练步骤和损失。 代码如下：

```python
import numpy as np
import tensorflow as tf
# Declare list of features, we only have one real-valued feature
def model(features, labels, mode):
  # Build a linear model and predict values
  W = tf.get_variable("W", [1], dtype=tf.float64)
  b = tf.get_variable("b", [1], dtype=tf.float64)
  y = W*features['x'] + b
  # Loss sub-graph
  loss = tf.reduce_sum(tf.square(y - labels))
  # Training sub-graph
  global_step = tf.train.get_global_step()
  optimizer = tf.train.GradientDescentOptimizer(0.01)
  train = tf.group(optimizer.minimize(loss),
                   tf.assign_add(global_step, 1))
  # ModelFnOps connects subgraphs we built to the
  # appropriate functionality.
  return tf.contrib.learn.ModelFnOps(
      mode=mode, predictions=y,
      loss=loss,
      train_op=train)

estimator = tf.contrib.learn.Estimator(model_fn=model)
# define our data sets
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])
input_fn = tf.contrib.learn.io.numpy_input_fn({"x": x_train}, y_train, 4, num_epochs=1000)

# train
estimator.fit(input_fn=input_fn, steps=1000)
# Here we evaluate how well our model did. 
train_loss = estimator.evaluate(input_fn=input_fn)
eval_loss = estimator.evaluate(input_fn=eval_input_fn)
print("train loss: %r"% train_loss)
print("eval loss: %r"% eval_loss)
```

When run, it produces

运行结果

```python
train loss: {'global_step': 1000, 'loss': 4.9380226e-11}
eval loss: {'global_step': 1000, 'loss': 0.01010081}
```

Notice how the contents of the custom `model()` function are very similar to our manual model training loop from the lower level API.

请注意这里的自定义model()函数和我们之前用底层API实现的手动训练模型很相似。