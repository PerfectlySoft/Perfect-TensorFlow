//
//  Package.swift
//  Perfect-TensorFlow
//
//  Created by Rockford Wei on 2017-05-18.
//  Copyright © 2017 PerfectlySoft. All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// This source file is part of the Perfect.org open source project
//
// Copyright (c) 2017 - 2018 PerfectlySoft Inc. and the Perfect project authors
// Licensed under Apache License v2.0
//
// See http://perfect.org/licensing.html for license information
//
//===----------------------------------------------------------------------===//
//
import PackageDescription

let package = Package(
    name: "PerfectTensorFlow",
    targets: [
      Target(name: "TensorFlowAPI", dependencies: []),
      Target(name: "PerfectTensorFlow", dependencies: ["TensorFlowAPI"])
    ],
    dependencies: [
      .Package(url: "https://github.com/apple/swift-protobuf.git", majorVersion:0)
    ]
)
