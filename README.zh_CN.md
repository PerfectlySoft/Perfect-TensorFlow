# Perfect TensorFlow
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

本项目为TensorFlow的C语言接口试验性封装函数库，用于Swift在人工智能深度学习上的应用。

本项目需要使用SPM软件包管理器编译并是[Perfect项目](https://github.com/PerfectlySoft/Perfect)的一个组成部分，但也可以独立使用。

请确保您的系统已经安装了Swift 3.1工具链。

## 项目状态

目前本函数库处于阿尔法测试状态，很快会有详细文档和更多案例支持。


## 快速上手

### TensorFlow API C语言库函数安装

Perfect-TensorFlow 是基于其C语言函数库基础上的，简单说来就是您的计算机上在运行时必须安装 `libtensorflow.so`动态链接库。

本项目包含了一个用于快速安装该链接库 CPU v1.1.0版本的脚本，默认安装路径为`/usr/local/lib/libtensorflow.so`。您可以根据平台要求下载并运行 [`install.sh`](https://github.com/PerfectlySoft/Perfect-TensorFlow/blob/master/install.sh)。

更多的安装选项，如需要在同一台计算机上同时安装CPU/GPU或者多个不同版本，请参考官网网站： [Installing TensorFlow for C](https://www.tensorflow.org/install/install_c)

### Perfect TensorFlow 应用程序

使用之前请在您的项目Package.swift文件中增加依存关系：

``` swift
.Package(url: "https://github.com/PerfectlySoft/Perfect-TensorFlow.git", majorVersion: 1)
```

然后声明函数库：

``` swift
/// TensorFlowAPI 就是定义在 libtensorflow.so的部分函数集
import TensorFlowAPI

/// 这是我们主要介绍的TensorFlow对象封装库
import PerfectTensorFlow

/// 为了保持与其他语言函数库版本（比如Python或者Java）的命名规范一致性，为TensorFlow对象取一个缩写名称是个好主意：
public typealias TF = TensorFlow
```

### 激活函数库

⚠️注意⚠️ 在使用  Perfect TensorFlow 的 **任何具体函数之前**，必须首先调用`TF.Open()`方法：

``` swift
/// 这个操作会打开 /usr/local/lib/libtensorflow.so 动态链接库
try TF.Open()
```

另外，您还可以激活其他不同规格（CPU/GPU）版本的函数库，所需要的操作就是输入目标函数库路径：
``` swift
/// 以下操作将打开非默认路径下的函数库：
try TF.Open("/path/to/DLL/of/libtensorflow.so")
```

### "你好，Perfect TensorFlow!"

以下是 Swift 版本的 "你好, TensorFlow!":

``` swift
// 定义一个字符串型张量：
let tensor = try TF.Tensor.Scalar("你好，Perfect TensorFlow! 🇨🇳🇨🇦")

// 声明一个流程图
let g = try TF.Graph()

// 将张量节点加入流程图
let op = try g.const(tensor: tensor, name: "hello")

// 根据流程图生成会话并运行
let o = try g.runner().fetch(op).addTarget(op).run()

// 解码
let decoded = try TF.Decode(strings: o[0].data, count: 1)

// 检查结果
let s2 = decoded[0].string
print(s2)
```

### 矩阵操作

您可以注意到，其实Swift版本的TensorFlow与其原版内容的概念都是完全一致的，比如创建张量节点，保存节点到流程图、定义操作并运行会话、最后检查结果。

以下是使用Perfect TensorFlow进行矩阵操作的例子：

``` swift
/* 矩阵乘法
| 1 2 |   |0 1|   |0 1|
| 3 4 | * |0 0| = |0 3|
*/
// 输入矩阵
// *注意* 不管目标矩阵的维度到底是多少，输入时必须按照扁平化处理，即所有元素一个挨一个列出
let srcA:[Float] = [[1, 2], [3, 4]].flatMap { $0 }
let srcB:[Float] = [[0, 0], [1, 0]].flatMap { $0 }

// 根据矩阵创建张量
let tA = try TF.Tensor.Array(dimenisons: [2,2], value: srcA)
let tB = try TF.Tensor.Array(dimenisons: [2,2], value: srcB)

// 将张量转化为流程图节点
let g = try TF.Graph()
let A = try g.const(tensor: tA, name: "Const_0")
let B = try g.const(tensor: tB, name: "Const_1")

// 定义矩阵乘法操作，即 v = A x Bt，B矩阵的转置
let v = try g.matMul(l: A, r: B, name: "v", transposeB: true)

// 运行会话
let o = try g.runner().fetch(v).addTarget(v).run()
let m:[Float] = try o[0].asArray()
print(m)
// m 的值应该是 [0, 1, 0, 3]
```

### 动态加载已保存的人工神经网络模型

除了动态建立流程图和会话方法之外，Perfect TensorFlow 还提供了将预先保存的模型在运行时加载的简单方法，即从文件中直接还原会话：

``` swift
let g = try TF.Graph()

// 读取模型的签名信息
let metaBuf = try TF.Buffer()

// 还原会话
let session = try g.load(
	exportDir: "/path/to/saved/model",
	tags: ["tag1", "tag2", ...],
	metaGraphDef: metaBuf)
```

详细例子请参考 [Perfect TensorFlow 测试用例](https://github.com/PerfectlySoft/Perfect-TensorFlow/blob/master/Tests/PerfectTensorFlowTests/PerfectTensorFlowTests.swift#L349-L390).


### 问题报告、内容贡献和客户支持

我们目前正在过渡到使用JIRA来处理所有源代码资源合并申请、修复漏洞以及其它有关问题。因此，GitHub 的“issues”问题报告功能已经被禁用了。

如果您发现了问题，或者希望为改进本文提供意见和建议，[请在这里指出](http://jira.perfect.org:8080/servicedesk/customer/portal/1).

在您开始之前，请参阅[目前待解决的问题清单](http://jira.perfect.org:8080/projects/ISS/issues).

## 更多信息
关于本项目更多内容，请参考[perfect.org](http://perfect.org).

## 扫一扫 Perfect 官网微信号
<p align=center><img src="https://raw.githubusercontent.com/PerfectExamples/Perfect-Cloudinary-ImageUploader-Demo/master/qr.png"></p>
