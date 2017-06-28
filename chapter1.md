# Getting Started With TensorFlow

# TensorFlow入门

This guide gets you started programming in TensorFlow. Before using this guide,[install TensorFlow](https://www.tensorflow.org/install/index). To get the most out of this guide, you should know the following:

本教程指导你如何用TensorFlow开始编程。在开始前，请先安装TensorFlow。为了更好的使用本教程，你还应该知道下面的知识：

* How to program in Python.
* 如何使用Python编程
* At least a little bit about arrays.
* 了解一些数组知识
* Ideally, something about machine learning. However, if you know little or nothing about machine learning, then this is still the first guide you should read.
* 如果还知道一些机器学习的知识就更好了。当然，如果你从未了解机器学习，本教程也是你应该阅读的第一个教程。

TensorFlow provides multiple APIs. The lowest level API--TensorFlow Core-- provides you with complete programming control. We recommend TensorFlow Core for machine learning researchers and others who require fine levels of control over their models. The higher level APIs are built on top of TensorFlow Core. These higher level APIs are typically easier to learn and use than TensorFlow Core. In addition, the higher level APIs make repetitive tasks easier and more consistent between different users. A high-level API like tf.contrib.learn helps you manage data sets, estimators, training and inference. Note that a few of the high-level TensorFlow APIs--those whose method names contain`contrib`-- are still in development. It is possible that some`contrib`methods will change or become obsolete in subsequent TensorFlow releases.

TensorFlow提供了多种API接口。最底层的API--TensorFlow Core-- 提供了完整的编程控制。我们推荐机器学习的研究人员和那些需要更好地控制他们的模型的人使用TensorFlow Core。有些更高级别的API比TensorFlow Core更容易学习和使用，而且，这些更高级别的API使不同的用户间的重复任务更容易实现及保持一致性。一个高层API如tf.contrib.learn将帮助你管理数据集，估算，训练和推论。值得注意的有少数几个高层TensorFlow API的名字包含了contrib --表示还在开发中。有可能这些contrib方法会在以后的TensorFlow版本中被修改或废除。

This guide begins with a tutorial on TensorFlow Core. Later, we demonstrate how to implement the same model in tf.contrib.learn. Knowing TensorFlow Core principles will give you a great mental model of how things are working internally when you use the more compact higher level API.

本教程从TensorFlow Core的入门指导开始，然后我们将演示如何用tf.contrib.learn去实现同样的模型。知道了TensorFlow Core的原理将会在你脑子里建立一个模型，这个模型会帮助你理解当你使用封装更好的高层API时，它们是如何工作的。

# Tensors {#tensors}

# 张量 {#tensors}

The central unit of data in TensorFlow is the **tensor**. A tensor consists of a set of primitive values shaped into an array of any number of dimensions. A tensor's **rank **is its number of dimensions. Here are some examples of tensors:

TensorFlow的核心数据是张量。张量是一组基本值构成的任意维度的数组。张量的阶指的是它的维度数。下面是一些张量的例子：

```
3 # a rank 0 tensor; this is a scalar with shape []
3 # 0阶张量；这是一个用形状[]来表示的标量
[1. ,2., 3.] # a rank 1 tensor; this is a vector with shape [3]
[1. ,2., 3.] # 1阶张量; 这是一个用形状[3]来表示的矢量
[[1., 2., 3.], [4., 5., 6.]] # a rank 2 tensor; a matrix with shape [2, 3]
[[[1., 2., 3.]], [[7., 8., 9.]]] # a rank 3 tensor with shape [2, 1, 3]
```

## TensorFlow Core tutorial {#tensorflow_core_tutorial}

### Importing TensorFlow {#importing_tensorflow}

The canonical import statement for TensorFlow programs is as follows:

```
import
 tensorflow 
as
 tf
```

This gives Python access to all of TensorFlow's classes, methods, and symbols. Most of the documentation assumes you have already done this.

