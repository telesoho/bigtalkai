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

TensorFlow提供了多种API接口。最底层的API--TensorFlow Core-- 提供了完整的编程控制。我们推荐机器学习的研究人员和那些需要更好地控制他们的模型的人使用TensorFlow Core。有些更高级别的API比TensorFlow Core更容易学习和使用，而且，这些更高级别的API使不同的用户间的重复任务更容易和保存一致性。

This guide begins with a tutorial on TensorFlow Core. Later, we demonstrate how to implement the same model in tf.contrib.learn. Knowing TensorFlow Core principles will give you a great mental model of how things are working internally when you use the more compact higher level API.

# Tensors {#tensors}

The central unit of data in TensorFlow is the**tensor**. A tensor consists of a set of primitive values shaped into an array of any number of dimensions. A tensor's**rank**is its number of dimensions. Here are some examples of tensors:

```
3 # a rank 0 tensor; this is a scalar with shape []
[1. ,2., 3.] # a rank 1 tensor; this is a vector with shape [3]
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

