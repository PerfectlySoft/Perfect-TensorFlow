# Perfect TensorFlow [简体中文](README.zh_CN.md)

<p align="center">
    <a href="http://perfect.org/get-involved.html" target="_blank">
        <img src="http://perfect.org/assets/github/perfect_github_2_0_0.jpg" alt="Get Involved with Perfect!" width="854" />
    </a>
</p>

<p align="center">
    <a href="https://github.com/PerfectlySoft/Perfect" target="_blank">
        <img src="http://www.perfect.org/github/Perfect_GH_button_1_Star.jpg" alt="Star Perfect On Github" />
    </a>  
    <a href="http://stackoverflow.com/questions/tagged/perfect" target="_blank">
        <img src="http://www.perfect.org/github/perfect_gh_button_2_SO.jpg" alt="Stack Overflow" />
    </a>  
    <a href="https://twitter.com/perfectlysoft" target="_blank">
        <img src="http://www.perfect.org/github/Perfect_GH_button_3_twit.jpg" alt="Follow Perfect on Twitter" />
    </a>  
    <a href="http://perfect.ly" target="_blank">
        <img src="http://www.perfect.org/github/Perfect_GH_button_4_slack.jpg" alt="Join the Perfect Slack" />
    </a>
</p>

<p align="center">
    <a href="https://developer.apple.com/swift/" target="_blank">
        <img src="https://img.shields.io/badge/Swift-3.0-orange.svg?style=flat" alt="Swift 3.0">
    </a>
    <a href="https://developer.apple.com/swift/" target="_blank">
        <img src="https://img.shields.io/badge/Platforms-OS%20X%20%7C%20Linux%20-lightgray.svg?style=flat" alt="Platforms OS X | Linux">
    </a>
    <a href="http://perfect.org/licensing.html" target="_blank">
        <img src="https://img.shields.io/badge/License-Apache-lightgrey.svg?style=flat" alt="License Apache">
    </a>
    <a href="http://twitter.com/PerfectlySoft" target="_blank">
        <img src="https://img.shields.io/badge/Twitter-@PerfectlySoft-blue.svg?style=flat" alt="PerfectlySoft Twitter">
    </a>
    <a href="http://perfect.ly" target="_blank">
        <img src="http://perfect.ly/badge.svg" alt="Slack Status">
    </a>
</p>

This project is an experimental wrapper of TensorFlow C API which enables Machine Learning in Server Side Swift.

This package builds with Swift Package Manager and is part of the [Perfect](https://github.com/PerfectlySoft/Perfect) project but can also be used as an independent module.

Ensure you have installed and activated the latest Swift 3.1 tool chain.

## Project Status

The framework is Alpha testing now. Documents and examples are coming soon.

## Quick Start

### TensorFlow C API Library Installation

Perfect-TensorFlow is based on TensorFlow C API, i.e., `libtensorflow.so` on runtime.
This project contains an express CPU v1.1.0 installation script for this module on both macOS / Ubuntu Linux, and will install the dynamic library into path `/usr/local/lib/libtensorflow.so`. You can download & run [`install.sh`](https://github.com/PerfectlySoft/Perfect-TensorFlow/blob/master/install.sh).

For more installation options, such as GPU/CPU and multiple versions on the same machine, please check TensorFlow website: [Installing TensorFlow for C](https://www.tensorflow.org/install/install_c)

### Perfect TensorFlow Application

To use this library, add dependencies to your project's Package.swift with the **LATEST TAG**:

``` swift
.Package(url: "https://github.com/PerfectlySoft/Perfect-TensorFlow.git", majorVersion: 1)
```

Then declare the library:

``` swift
// TensorFlowAPI contains most API functions defined in libtensorflow.so
import TensorFlowAPI

// This is the Swift version of TensorFlow classes and objects
import PerfectTensorFlow

// To keep the naming consistency with TensorFlow in other languages such as 
// Python or Java, making an alias of `TensorFlow` Class is a good idea:
public typealias TF = TensorFlow
```

### Library Activation

⚠️NOTE⚠️ Prior to use **ANY ACTUAL FUNCTIONS** of Perfect TensorFlow framework, `TF.Open()` must be called first:

``` swift
// this action will load all api functions defined 
// in /usr/local/lib/libtensorflow.so
try TF.Open()
```

Please also note that you can active the library with a specific path, alternatively, especially in case of different versions or CPU/GPU library adjustment required:

``` swift
// this action will load the library with the path
try TF.Open("/path/to/DLL/of/libtensorflow.so")
```

### "Hello, Perfect TensorFlow!"

Here is the Swift version of "Hello, TensorFlow!":

``` swift
// define a string tensor
let tensor = try TF.Tensor.Scalar("Hello, Perfect TensorFlow! 🇨🇳🇨🇦")

// declare a new graph
let g = try TF.Graph()

// turn the tensor into an operation
let op = try g.const(tensor: tensor, name: "hello")

// run a session
let o = try g.runner().fetch(op).addTarget(op).run()

// decode the result      
let decoded = try TF.Decode(strings: o[0].data, count: 1)

// check the result
let s2 = decoded[0].string
print(s2)
```

### Matrix Operations


As you can see, Swift version of TensorFlow keeps the same principals of the original one, i.e., create tensors, save tensors into graph, define the operations and then run the session & check the result.

Here is an other simple example of matrix operations in Perfect TensorFlow:

``` swift
/* Matrix Muliply:
| 1 2 |   |0 1|   |0 1|
| 3 4 | * |0 0| = |0 3|
*/
// input the matrix.
// *NOTE* no matter how many dimensions a matrix may have,
// the matrix should always input as an flattened array
let srcA:[Float] = [[1, 2], [3, 4]].flatMap { $0 }
let srcB:[Float] = [[0, 0], [1, 0]].flatMap { $0 }

// create tensors for these matrics
let tA = try TF.Tensor.Array(dimenisons: [2,2], value: srcA)
let tB = try TF.Tensor.Array(dimenisons: [2,2], value: srcB)

// adding tensors to graph
let g = try TF.Graph()
let A = try g.const(tensor: tA, name: "Const_0")
let B = try g.const(tensor: tB, name: "Const_1")

// define matrix multiply operation
let v = try g.matMul(l: A, r: B, name: "v", transposeB: true)

// run the session
let o = try g.runner().fetch(v).addTarget(v).run()
let m:[Float] = try o[0].asArray()
print(m)
// m shall be [0, 1, 0, 3]
```

### Load a Saved Artificial Neural Network Model

Besides building graph & sessions in code, Perfect TensorFlow also provides a handy method to load models into runtime, i.e, generate a new session by loading a model file:

``` swift
let g = try TF.Graph()

// the meta signature info defined in a saved model
let metaBuf = try TF.Buffer()

// load the session
let session = try g.load(
	exportDir: "/path/to/saved/model",
	tags: ["tag1", "tag2", ...],
	metaGraphDef: metaBuf)
```

A detailed example of loading model can be found in the [Perfect TensorFlow Testing Examples](https://github.com/PerfectlySoft/Perfect-TensorFlow/blob/master/Tests/PerfectTensorFlowTests/PerfectTensorFlowTests.swift#L349-L390).

## Issues

We are transitioning to using JIRA for all bugs and support related issues, therefore the GitHub issues has been disabled.

If you find a mistake, bug, or any other helpful suggestion you'd like to make on the docs please head over to [http://jira.perfect.org:8080/servicedesk/customer/portal/1](http://jira.perfect.org:8080/servicedesk/customer/portal/1) and raise it.

A comprehensive list of open issues can be found at [http://jira.perfect.org:8080/projects/ISS/issues](http://jira.perfect.org:8080/projects/ISS/issues)

## Further Information
For more information on the Perfect project, please visit [perfect.org](http://perfect.org).


## Now WeChat Subscription is Available (Chinese)
<p align=center><img src="https://raw.githubusercontent.com/PerfectExamples/Perfect-Cloudinary-ImageUploader-Demo/master/qr.png"></p>
