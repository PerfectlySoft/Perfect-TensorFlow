// swift-tools-version:4.0
// The swift-tools-version declares the minimum version of Swift required to build this package.
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
#if os(OSX)
import Darwin
#else
import Glibc
#endif
let package = Package(
    name: "PerfectTensorFlow",
    products: [
        .library(
            name: "PerfectTensorFlow",
            targets: ["PerfectTensorFlow"]),
    ],
    dependencies: [
      .package(url: "https://github.com/apple/swift-protobuf.git", from: "1.0.0")
    ],
    targets: [
        .target(
            name: "TensorFlowAPI",
            dependencies: []),
        .target(
            name: "PerfectTensorFlow",
            dependencies: ["TensorFlowAPI", "SwiftProtobuf"],
            exclude:[]),
        .testTarget(
            name: "PerfectTensorFlowTests",
            dependencies: ["PerfectTensorFlow"]),
    ]
)
