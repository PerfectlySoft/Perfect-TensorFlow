//
//  PerfectTensorFlowTests.swift
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

import XCTest
@testable import PerfectTensorFlow
import TensorFlowAPI


public typealias TF = TensorFlow

struct SavedModel {
  /// SavedModel assets directory.
  public static let kSavedModelAssetsDirectory  = "assets"

  /// SavedModel assets key for graph collection-def.
  public static let kSavedModelAssetsKey  = "saved_model_assets"

  /// SavedModel proto filename.
  public static let kSavedModelFilenamePb  = "saved_model.pb"

  /// SavedModel text format proto filename.
  public static let kSavedModelFilenamePbTxt  = "saved_model.pbtxt"

  /// SavedModel legacy init op key.
  public static let kSavedModelLegacyInitOpKey  = "legacy_init_op"

  /// SavedModel main op key.
  public static let kSavedModelMainOpKey  = "saved_model_main_op"

  /// Directory in which to save the SavedModel variables.
  public static let kSavedModelVariablesDirectory  = "variables"

  /// SavedModel variables filename.
  public static let kSavedModelVariablesFilename  = "variables"

  /// Key in the signature def map for `default` serving signatures. The default
  /// signature is used in inference requests where a specific signature was not
  /// specified.
  public static let kDefaultServingSignatureDefKey  = "serving_default"

  ////////////////////////////////////////////////////////////////////////////////
  /// Classification API constants.

  /// Classification inputs.
  public static let kClassifyInputs  = "inputs"

  /// Classification method name used in a SignatureDef.
  public static let kClassifyMethodName  = "tensorflow/serving/classify"

  /// Classification classes output.
  public static let kClassifyOutputClasses  = "classes"

  /// Classification scores output.
  public static let kClassifyOutputScores  = "scores"

  ////////////////////////////////////////////////////////////////////////////////
  /// Predict API constants.

  /// Predict inputs.
  public static let kPredictInputs  = "inputs"

  /// Predict method name used in a SignatureDef.
  public static let kPredictMethodName  = "tensorflow/serving/predict"

  /// Predict outputs.
  public static let kPredictOutputs  = "outputs"

  ////////////////////////////////////////////////////////////////////////////////
  /// Regression API constants.

  /// Regression inputs.
  public static let kRegressInputs  = "inputs"

  /// Regression method name used in a SignatureDef.
  public static let kRegressMethodName  = "tensorflow/serving/regress"

  /// Regression outputs.
  public static let kRegressOutputs  = "outputs"

  /// Tag for the `serving` graph.
  public static let kSavedModelTagServe  = "serve"

  /// Tag for the `training` graph.`
  public static let kSavedModelTagTrain  = "train"
}//end struct

public extension Data {
  public static func Load(_ localFile: String) -> Data? {
    var st = stat()
    guard let f = fopen(localFile, "rb"), stat(localFile, &st) == 0, st.st_size > 0 else { return nil }
    let size = Int(st.st_size)
    let buf = UnsafeMutablePointer<UInt8>.allocate(capacity: size)
    guard size == fread(buf, 1, size, f) else {
      buf.deallocate(capacity: size)
      return nil
    }//end guard
    return Data(bytesNoCopy: buf, count: size, deallocator: .free)
  }
}

class PerfectTensorFlowTests: XCTestCase {

  static var allTests = [
    ("testVersion", testVersion),
    ("testSize", testSize),
    ("testStatus", testStatus),
    ("testBuffer", testBuffer),
    ("testTensorScalarConst", testTensorScalarConst),
    ("testSessionOptions", testSessionOptions),
    ("testGraph", testGraph),
    ("testGraph2", testGraph2),
    ("testImportGraphDef", testImportGraphDef),
    ("testOpList", testOpList),
    ("testSetShapePlaceHolder", testSetShapePlaceHolder),
    ("testSetShape", testSetShape),
    ("testSession", testSession),
    ("testSessionExpress", testSessionExpress),
    ("testPSession", testPSession),
    ("testShapeInference", testShapeInference),
    ("testSavedModel", testSavedModel),
    ("testBasicLoop", testBasicLoop),
    ("testAttributes", testAttributes),
    ("testEncodeDecode", testEncodeDecode),
    ("testHelloWorld", testHelloWorld),
    ("testHelloWorldUTF8", testHelloWorldUTF8),
    ("testHelloExpress", testHelloExpress),
    ("testBasic", testBasic),
    ("testBasicExpress", testBasicExpress),
    ("testLabels", testLabels),
    ("testSessionLeak", testSessionLeak),
    ("testGradients", testGradients),
    ("testMatrix", testMatrix),
    ("testBasicImproved",testBasicImproved),
    ("testDevices", testDevices)
  ]

  func testDevices() {
    do {
      let g = try TF.Graph()
      let dev = try g.runner().session.devices
      print("default devices:", dev)
      XCTAssertGreaterThan(dev.count, 0)
    }catch {
      XCTFail("hello: \(error)")
    }

  }

  func testMatrix() {
    let x = [[1, 2, 3], [4, 5, 6]]
    let y = [[[1,2],[3,4],[5,6]],[[7,8],[9,10],[11,12]],[[13,14],[15,16],[17,18]],[[19,20],[21,22],[23,24]]]
    XCTAssertEqual(x.shape, [2, 3])
    XCTAssertEqual(y.shape, [4, 3, 2])
    XCTAssertEqual(x[1][2], 6)
    XCTAssertEqual(x.column(index: 1) as! [Int], [2, 5])
  }

  func testGradients() {
    do {
      let grad = TestGradients()
      try grad.test(true)
      try grad.test(false)
    }catch {
      XCTFail("gradients: \(error)")
    }
  }

  class TestGradients {

    public func test(_ providingDX: Bool) throws {
      let success = try buildSuccessGraph()
      let expected = try buildExpectedGraph(providingDX)
      let _ = try addGradients(providingDX, graph: success.graph, inputs: success.inputs, outputs: success.outputs)

      guard let def0 = success.graph.definition,
        let def1 = expected.graph.definition else {
        throw TF.Panic.FAULT(reason: "Unexpected Graph Definition after Adding Gradients")
      }

      let n0 = def0.node
      let n1 = def1.node
      XCTAssertEqual(n0.count, n1.count)
      for i in 0 ..< n0.count {
        let n = n0[i]
        let m = n1[i]
        //XCTAssertEqual(n.name, m.name)
        XCTAssertEqual(n.op, m.op)
        for key in n.attr.keys {
          let v = n.attr[key]
          let w = m.attr[key]
          XCTAssertEqual(v, w)
        }
      }

    }
    public func buildSuccessGraph() throws ->
      (graph: TF.Graph, inputs:[TF.Output], outputs: [TF.Output])
    {
      // Construct the following graph:
      //            |
      //           z|
      //            |
      //          MatMul
      //         /       \
      //        ^         ^
      //        |         |
      //       x|        y|
      //        |         |
      //        |         |
      //      Const_0    Const_1
      //
      let srcA:[[Float]] = [[1,2],[3,4]]
      let srcB:[[Float]] = [[1,0],[0,1]]

      // create tensors for these matrices
      let tA = try TF.Tensor.Matrix(srcA)
      let tB = try TF.Tensor.Matrix(srcB)

      let g = try TF.Graph()
      // adding tensors to graph
      let A = try g.const(tensor: tA, name: "Const_0")
      let B = try g.const(tensor: tB, name: "Const_1")
      let M = try g.matMul(l: A, r: B, name: "MatMul")
      return (graph: g, inputs:[A.asOutput(0), B.asOutput(0)], outputs:[M.asOutput(0)])
    }

    public func buildExpectedGraph(_ providingDX: Bool) throws ->
      (graph: TF.Graph, dy: [TF.Output])
    {
      // The expected graph looks like this if grad_inputs_provided.
      // If grad_inputs_provided is false, Const_0 will be a OnesLike op.
      //      ^             ^
      //    dy|           dx|        // MatMul Gradient Graph
      //      |             |
      //   MatMul_2      MatMul_1
      //   ^   ^          ^    ^
      //   |   |----------|    |
      //   |        ^          |
      //   |      dz|          |
      //   |        |          |
      //   |     Const_3       |
      //   |                   |
      //   |        ^          |
      //   |       z|          |     // MatMul Forward Graph
      //   |        |          |
      //   |      MatMul       |
      //   |     /       \     |
      //   |    ^         ^    |
      //   |    |         |    |
      //   |---x|        y|----|
      //        |         |
      //        |         |
      //      Const_0   Const_1
      let srcA:[[Float]] = [[1,2],[3,4]]
      let srcB:[[Float]] = [[1,0],[0,1]]

      // create tensors for these matrices
      let tA = try TF.Tensor.Matrix(srcA)
      let tB = try TF.Tensor.Matrix(srcB)

      let g = try TF.Graph()
      // adding tensors to graph
      let A = try g.const(tensor: tA, name: "Const_0")
      let B = try g.const(tensor: tB, name: "Const_1")
      let M = try g.matMul(l: A, r: B, name: "MatMul")

      let C: TF.Operation
      if providingDX {
        let srcC: [[Float]] = [[1, 1], [1, 1]]
        let tC = try TF.Tensor.Matrix(srcC)
        C = try g.const(tensor: tC, name: "GradInputs")
      } else {
        C = try g.OnesLike(inp: M, name: "OnesLike")
      }//end if

      let M1 = try g.matMul(l: C, r: B, name: "MatMul_1", transposeA: false, transposeB: true)
      let M2 = try g.matMul(l: A, r: C, name: "MatMul_2", transposeA: true, transposeB: false)
      return (graph: g, dy: [M1.asOutput(0), M2.asOutput(0)])
    }

    public func addGradients(_ providingDX: Bool, graph: TF.Graph, inputs: [TF.Output], outputs: [TF.Output]) throws -> [TF.Output] {
      if providingDX {
        let dxArray:[[Float]] = [[1, 1], [1, 1]]
        let dxValue = try TF.Tensor.Matrix(dxArray)
        let dx = try graph.const(tensor: dxValue, name: "GradInputs").asOutput(0)
        return try graph.addGradients(y: outputs, x: inputs, dx: [dx])
      } else {
        return try graph.addGradients(y: outputs, x: inputs)
      }
    }
  }
  func testLabels() {
    do {
      let img = try LabelImage()
      guard let eight = Data.Load("/tmp/testdata/8.jpg") else
      { throw TF.Panic.FAULT(reason: "hand write file 8.jpg not found")}
      for _ in 0 ... 10 {
        #if os(Linux)
          let x = try img.match(image: eight)
          XCTAssertEqual(x, 536)
        #else
          autoreleasepool(invoking: {
            do {
              let x = try img.match(image: eight)
              XCTAssertEqual(x, 536)
            }catch {
              XCTFail("label loop: \(error)")
            }
          })
        #endif
      }
    }catch {
      XCTFail("label: \(error)")
    }
  }

  class LabelImage {
    let def: TF.GraphDef

    public init(_ modelPath:String = "/tmp/testdata/tensorflow_inception_graph.pb") throws {
      guard let bytes = Data.Load(modelPath) else { throw TF.Panic.INVALID }
      def = try TF.GraphDef(serializedData: bytes)

    }

    public func match(image: Data) throws -> Int {
      let g = try TF.Graph()
      try g.import(definition: def)
      let normalized = try constructAndExecuteGraphToNormalizeImage(g, imageBytes: image)
      let possibilities = try executeInceptionGraph(g, image: normalized)
      guard let m = possibilities.max(), let i = possibilities.index(of: m) else {
        throw TF.Panic.INVALID
      }//end guard
      return i
    }

    private func executeInceptionGraph(_ g: TF.Graph, image: TF.Tensor) throws -> [Float] {
      let results = try g.runner().feed("input", tensor: image).fetch("output").run()
      guard results.count > 0 else { throw TF.Panic.INVALID }
      let result = results[0]
      guard result.dimensionCount == 2 else { throw TF.Panic.INVALID }
      let shape = result.dim
      guard shape[0] == 1 else { throw TF.Panic.INVALID }
      let res: [Float] = try result.asArray()
      return res
    }//end exec

    public func constructAndExecuteGraphToNormalizeImage(_ g: TF.Graph, imageBytes: Data) throws -> TF.Tensor{
      let H:Int32 = 224
      let W:Int32 = 224
      let mean:Float = 117
      let scale:Float = 1
      let input = try g.constant(name: "input2", value: imageBytes)
      let batch = try g.constant( name: "make_batch", value: Int32(0))
      let scale_v = try g.constant(name: "scale", value: scale)
      let mean_v = try g.constant(name: "mean", value: mean)
      let size = try g.constantArray(name: "size", value: [H,W])
      let jpeg = try g.decodeJpeg(content: input, channels: 3)
      let cast = try g.cast(value: jpeg, dtype: TF.DataType.dtFloat)
      let images = try g.expandDims(input: cast, dim: batch)
      let resizes = try g.resizeBilinear(images: images, size: size)
      let subbed = try g.sub(x: resizes, y: mean_v)
      let output = try g.div(x: subbed, y: scale_v)
      let s = try g.runner().fetch(TF.Operation(output)).run()
      guard s.count > 0 else { throw TF.Panic.INVALID }
      return s[0]
    }//end normalize

  }

  func testBasicImproved() {
    do {
      /*
       Matrix Test:
       | 1 2 |  |0 1|  |0 1|
       |     |* |   |= |   |
       | 3 4 |  |0 0|  |0 3|
       */
      let tA = try TF.Tensor.Matrix([[1, 2], [3, 4]])
      let tB = try TF.Tensor.Matrix([[0, 0], [1, 0]])
      let g = try TF.Graph()
      let A = try g.const(tensor: tA, name: "Const_0")
      let B = try g.const(tensor: tB, name: "Const_1")
      let v = try g.matMul(l: A, r: B, name: "v", transposeB: true)
      let o = try g.runner().fetch(v).addTarget(v).run()
      let m:[Float] = try o[0].asArray()
      let r:[Float] = [0, 1, 0, 3]
      XCTAssertEqual(m, r)
    }catch {
      XCTFail("improved: \(error)")
    }
  }

  func testBasicExpress() {
    do {
      /*
       Matrix Test:
       | 1 2 |  |0 1|  |0 1|
       |     |* |   |= |   |
       | 3 4 |  |0 0|  |0 3|
       */
      let srcA:[Float] = [[1, 2], [3, 4]].flatMap { $0 }
      let srcB:[Float] = [[0, 0], [1, 0]].flatMap { $0 }
      let tA = try TF.Tensor.Array(dimensions: [2,2], value: srcA)
      let tB = try TF.Tensor.Array(dimensions: [2,2], value: srcB)
      let tgtA:[Float] = try tA.asArray()
      let tgtB:[Float] = try tB.asArray()
      XCTAssertEqual(srcA, tgtA)
      XCTAssertEqual(srcB, tgtB)
      let g = try TF.Graph()
      let A = try g.const(tensor: tA, name: "Const_0")
      let B = try g.const(tensor: tB, name: "Const_1")
      let v = try g.matMul(l: A, r: B, name: "v", transposeB: true)
      let o = try g.runner().fetch(v).addTarget(v).run()
      let m:[Float] = try o[0].asArray()
      let r:[Float] = [0, 1, 0, 3]
      XCTAssertEqual(m, r)
    }catch {
      XCTFail("basic: \(error)")
    }
  }

  func testBasic() {
    do {
      /*
       Matrix Test:
       | 1 2 |  |0 1|  |0 1|
       |     |* |   |= |   |
       | 3 4 |  |0 0|  |0 3|
       */
      let srcA:[Float] = [1, 2, 3, 4]
      let srcB:[Float] = [0, 0, 1, 0]
      let tA = try TF.Tensor.Array(dimensions: [2,2], value: srcA)
      let tB = try TF.Tensor.Array(dimensions: [2,2], value: srcB)
      let tgtA:[Float] = try tA.asArray()
      let tgtB:[Float] = try tB.asArray()
      XCTAssertEqual(srcA, tgtA)
      XCTAssertEqual(srcB, tgtB)
      let g = try TF.Graph()
      let A = try g.const(tensor: tA, name: "Const_0")
      let B = try g.const(tensor: tB, name: "Const_1")
      let v = try g.matMul(l: A, r: B, name: "v", transposeB: true)
      let sess = try g.newSession()
      let o = try sess.run(outputs: [v.output(0)], targets:[v])
      let m:[Float] = try o[0].asArray()
      let r:[Float] = [0, 1, 0, 3]
      XCTAssertEqual(m, r)
    }catch {
      XCTFail("basic: \(error)")
    }
  }

  func testSessionLeak() {
    let hello = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyz"
    for _ in 0 ... 100 {
      #if os(Linux)
        do {
          let g = try TF.Graph()
          let tensor = try TF.Tensor.Scalar(hello)
          let op = try g.const(tensor: tensor, name: "hello")
          let o = try g.runner().fetch(op).addTarget(op).run()
          let data = o[0].data
          let decoded = try TF.Decode(strings: data, count: 1)
          let s2 = decoded[0].string
          XCTAssertEqual(hello, s2)
        }catch {
          XCTFail("hello: \(error)")
        }
      #else
        autoreleasepool(invoking: {
          do {
            let g = try TF.Graph()
            let tensor = try TF.Tensor.Scalar(hello)
            let op = try g.const(tensor: tensor, name: "hello")
            let o = try g.runner().fetch(op).addTarget(op).run()
            let data = o[0].data
            let decoded = try TF.Decode(strings: data, count: 1)
            let s2 = decoded[0].string
            XCTAssertEqual(hello, s2)
          }catch {
            XCTFail("hello: \(error)")
          }
        })
      #endif
    }
  }

  func testHelloExpress() {
    do {
      let hello = "你好！ 完美的 TensorFlow! 🇨🇳🇨🇦"
      let g = try TF.Graph()
      let tensor = try TF.Tensor.Scalar(hello)
      let op = try g.const(tensor: tensor, name: "hello")
      let o = try g.runner().fetch(op).addTarget(op).run()
      let data = o[0].data
      let decoded = try TF.Decode(strings: data, count: 1)
      let s2 = decoded[0].string
      XCTAssertEqual(hello, s2)
    }catch {
      XCTFail("hello: \(error)")
    }
  }

  func testHelloWorldUTF8 () {
    do {
      let hello = "你好！ 完美的 TensorFlow! 🇨🇳🇨🇦"
      let g = try TF.Graph()
      let tensor = try TF.Tensor.Scalar(hello)
      let op = try g.const(tensor: tensor, name: "hello")
      let s = try g.newSession()
      let o = try s.run(outputs: [op.output(0)], targets: [op])
      let data = o[0].data
      let decoded = try TF.Decode(strings: data, count: 1)
      let s2 = decoded[0].string
      XCTAssertEqual(hello, s2)
    }catch {
      XCTFail("hello: \(error)")
    }
  }

  func testHelloWorld () {
    do {
      let hello = "Hello TensorFlow!"
      let g = try TF.Graph()
      let tensor = try TF.Tensor.Scalar(hello)
      let op = try g.const(tensor: tensor, name: "hello")
      let s = try g.newSession()
      let o = try s.run(outputs: [op.output(0)], targets: [op])
      let data = o[0].data
      let decoded = try TF.Decode(strings: data, count: 1)
      let s2 = decoded[0].string
      XCTAssertEqual(hello, s2)
    }catch {
      XCTFail("hello: \(error)")
    }
  }

  class CWhileLoopTest {
    let status: TF.Status
    let graph: TF.Graph
    var inputs_ :[TF.Output] = []
    var outputs_ :[TF.Output] = []
    let params_: TF.GraphWhile
    var original_graph_description_ = ""
    let session: TF.Session
    var output_tensors:[TF.Tensor] = []
    public init(ninputs: Int) throws {
      status = try TF.Status()
      graph = try TF.Graph()
      session = try graph.newSession()
      guard ninputs > 0 else { throw TF.Panic.INVALID }

      for i in 0 ... ninputs - 1 {
        let placeholder = try graph.placeholder(name: "p\(i)")
        inputs_.append(placeholder.output(0))
      }//next

      params_ = try TF.GraphWhile(graph: graph, inputs: inputs_)
      params_.param.name = UnsafePointer<CChar>(strdup("test_loop"))

      original_graph_description_ = self.graphDebugString
    }//init

    var graphDebugString: String {
      guard let def = graph.definition else { return "" }
      return def.debugDescription
    }

    func expectOK() -> Bool {
      do {
        outputs_ = try params_.finish()
        return true
      }catch {
        return false
      }
    }

    func expectError(msg: String) -> Bool {
      do {
        _ = try params_.finish()
        return false
      } catch TF.Panic.FAULT(reason: let rs) {
        return msg == rs
      } catch {
        return false
      }
    }

    func run(input_values: [Int]) {
      do {
        XCTAssertEqual(inputs_.count, input_values.count)
        var inputs:[(TF.Output, TF.Tensor)] = []
        for i in 0 ... inputs_.count - 1 {
          let op = TF.Operation(inputs_[i].oper)
          let tensor = try TF.Tensor.Scalar(Int32(input_values[i]))
          inputs.append((op.output(0), tensor))
        }//next

        output_tensors = try session.run(inputs: inputs, outputs: outputs_)
      }catch {
        XCTFail("CWhileLoopTest:\(error)")
      }
    }//end fun

    func expectOutput(idx: Int, value: Int) -> Bool {
      do {
        XCTAssertGreaterThan(idx, -1)
        XCTAssertGreaterThan(output_tensors.count, idx)
        let tensor = output_tensors[idx]
        XCTAssertEqual(tensor.type ?? TF.DataType.dtInvalid, TF.DataType.dtInt32)
        XCTAssertEqual(tensor.dimensionCount, 0)
        XCTAssertEqual(MemoryLayout<Int32>.size, tensor.bytesCount)
        let array: [Int32] = try tensor.asArray()
        XCTAssertEqual(array[0], Int32(value))
        return true
      }catch {
        XCTFail("CWhileLoopTest:\(error)")
        return false
      }
    }

    func createConGraph() {
      do {
        let one = try TF.Graph(handle: params_.param.cond_graph).scalar(1)
        let less_than = try self.graph.lessThan(left: params_.param.cond_inputs.pointee, right: one.output(0))
        params_.param.cond_output = less_than.output(0)
      }catch {
        XCTFail("CWhileLoopTest:\(error)")
      }
    }
  }

  func testBasicLoop() {
    do {
      let loop = try CWhileLoopTest(ninputs: 2)
      XCTAssertNotNil(loop.params_.param.body_graph)
      XCTAssertNotNil(loop.params_.param.cond_graph)
      XCTAssertEqual(2, loop.params_.param.ninputs)
      XCTAssertNotNil(loop.params_.param.cond_inputs)
      XCTAssertNotNil(loop.params_.param.cond_inputs.advanced(by: 0))
      XCTAssertNotNil(loop.params_.param.cond_inputs.advanced(by: 1))
      XCTAssertNotNil(loop.params_.param.body_outputs)
      let cond_graph = TF.Graph(handle: loop.params_.param.cond_graph)
      let body_graph = TF.Graph(handle: loop.params_.param.body_graph)
      let less_than = try cond_graph.lessThan(left: loop.params_.param.cond_inputs.advanced(by: 0).pointee, right: loop.params_.param.cond_inputs.advanced(by: 1).pointee)
      loop.params_.param.cond_output = less_than.output(0)
      let add1 = try body_graph.add(left: loop.params_.param.body_inputs.pointee, right: loop.params_.param.body_inputs.advanced(by: 1).pointee, name: "add1")
      let one = try body_graph.scalar(1)
      let add2 = try body_graph.add(left: add1, right: one, name: "add2")
      loop.params_.param.body_outputs.pointee = add2.output(0)
      loop.params_.param.body_outputs.advanced(by: 1).pointee = loop.params_.param.body_inputs.advanced(by: 1).pointee
      XCTAssertTrue(loop.expectOK())
      let o = loop.outputs_
      let o0 = o[0]
      let o1 = o[1]
      XCTAssertNotNil(o0.oper)
      XCTAssertGreaterThanOrEqual(o0.index, 0)
      XCTAssertNotNil(o1.oper)
      XCTAssertGreaterThanOrEqual(o1.index, 0)
      loop.run(input_values: [-9,2])
      XCTAssertTrue(loop.expectOutput(idx: 0, value: 3))
      XCTAssertTrue(loop.expectOutput(idx: 1, value: 2))
    }catch {
      XCTFail("basic loop: \(error)")
    }
  }

  class AttrTest {
    public init(_ dataType: TF.DataType, value: Any) {
      do {
        let graph = try TF.Graph()
        let builder = try graph.opBuilder(name: "feed", type: "Placehodler")
        _ = try builder.set(attributes: ["attr": value, "dtype": dataType])
      }catch{
        XCTFail("attributes (\(value)): \(error)")
      }
    }
  }

  func testAttributes () {
    _ = AttrTest(TF.DataType.dtInt64, value: Int64(0))
    _ = AttrTest(TF.DataType.dtInt64, value: [Int64(0), Int64(1)])
    _ = AttrTest(TF.DataType.dtFloat, value: 1.1)
    _ = AttrTest(TF.DataType.dtFloat, value: [1.1, 2.2])
    _ = AttrTest(TF.DataType.dtBool, value: true)
    _ = AttrTest(TF.DataType.dtBool, value: [true, false])
    _ = AttrTest(TF.DataType.dtString, value: "hello")
    _ = AttrTest(TF.DataType.dtString, value: ["hello", "world"])
  }//end func

  func testSavedModel () {
    do {
      let graph = try TF.Graph()
      let metaBuf = try TF.Buffer()

      let runner = try graph.load(exportDir: "/tmp/testdata/half_plus_two/00000123", tags: [SavedModel.kSavedModelTagServe], metaGraphDef: metaBuf)

      guard let data = metaBuf.data else {
        XCTFail("saved model: no data")
        return
      }
      let meta = try TF.MetaGraphDef(serializedData: data)
      guard let signature_def = meta.signatureDef["regress_x_to_y"] else {
        XCTFail("saved model: bad signature")
        return
      }
      guard
        let input_name = signature_def.inputs[SavedModel.kRegressInputs]?.name,
        let output_name = signature_def.outputs[SavedModel.kRegressOutputs]?.name
        else {
        XCTFail("saved model: bad signature name")
        return
      }
      XCTAssertEqual(input_name, "tf_example:0")
      XCTAssertEqual(output_name, "y:0")

      var dataArray = [Data]()
      for i in 0 ... 3 {
        var example = Tensorflow_Example()
        var fList = Tensorflow_FloatList()
        fList.value.append(Float(i))
        var feature = Tensorflow_Feature()
        feature.floatList = fList
        example.features.feature["x"] = feature
        let dat = try example.serializedData()
        dataArray.append(dat)
      }//next

      let input_op = try graph.searchOperation(forName: "tf_example").output(0)
      let input_op_value = try TF.Tensor.Array(dimensions: [Int64(4)], value: dataArray)

      let output_op = try graph.searchOperation(forName: "y").output(0)


      let outArray = try runner
        .feed(input_op, tensor: input_op_value)
        .fetch(output_op)
        .run()

      XCTAssertGreaterThan(outArray.count, 0)


      let out = outArray[0]

      XCTAssertEqual(out.type ?? TF.DataType.dtInvalid, TF.DataType.dtFloat)
      XCTAssertEqual(2, out.dimensionCount)
      XCTAssertEqual(4, try out.dimension(0))
      XCTAssertEqual(1, try out.dimension(1))
      let value:[Float] = try out.asArray()
      let expected : [Float] = [2, 2.5, 3, 3.5]
      XCTAssertEqual(value, expected)

    }catch {
      XCTFail("saved model: \(error)")
    }
  }

  func testShapeInference() {
    do {
      let graph = try TF.Graph()
      let vec2Tensor = try TF.Tensor.Array(dimensions: [Int64(2)], value: [Int8(1), Int8(2)])
      let vec3Tensor = try TF.Tensor.Array(dimensions: [Int64(3)], value: [Int8(1), Int8(2), Int8(3)])
      let vec2 = try graph.const(tensor: vec2Tensor, name: "vec2")
      let vec3 = try graph.const(tensor: vec3Tensor, name: "vec3")

      let x = try graph.add(left: vec2, right: vec3)
      XCTAssertNil(x)
    }catch {
      XCTAssertNotNil(error)
    }
  }//end test

  func testPSession() {
    do {
      let graph = try TF.Graph()
      let a = try graph.placeholder(name: "A")
      let b = try graph.placeholder(name: "B")
      let two = try graph.scalar(2)
      let plus2 = try graph.add(left: a, right: two, name: "plus2")
      let plusB = try graph.add(left: plus2, right: b, name: "plusB")

      let sess = try graph.newSession()
      let feeds = [a.output(0), b.output(0)]
      let fetches = [plus2.output(0), plusB.output(0)]

      let handle = try sess.partial(inputs: feeds, outputs: fetches)

      let oneTensor = try TF.Tensor.Scalar(Int32(1))

      let feeds1 = [(input: a.output(0), tensor: oneTensor)]
      let fetches1 = [ plus2.output(0)]

      let fetchValues1 = try handle.run(inputs: feeds1, outputs: fetches1)
      let out1Value: Int32 = try fetchValues1[0].asArray()[0]
      XCTAssertEqual(3, out1Value)

      let fourTensor = try TF.Tensor.Scalar(Int32(4))

      let feeds2 = [(input: b.output(0), tensor: fourTensor)]
      let fetches2 = [plusB.output(0)]
      let fetchValues2 = try handle.run(inputs: feeds2, outputs: fetches2)
      let out2Value: Int32 = try fetchValues2[0].asArray()[0]
      XCTAssertEqual(7, out2Value)

    }catch {
      XCTFail("partial: \(error)")
    }
  }

  func testSessionExpress() {
    do {
      let graph = try TF.Graph()
      let feed = try graph.placeholder()
      let two = try graph.scalar(2)
      let add = try graph.add(left: feed, right: two)
      let threeTensor = try TF.Tensor.Scalar(Int32(3))

      let outs = try graph.runner().feed(feed, tensor: threeTensor).fetch(add).run()

      XCTAssertFalse(outs.isEmpty)
      let out = outs[0]
      XCTAssertEqual(out.type ?? TF.DataType.dtInvalid, TF.DataType.dtInt32)
      XCTAssertEqual(0, out.dimensionCount)
      XCTAssertEqual(MemoryLayout<Int32>.size, out.bytesCount)
      let outValue:Int32 = try out.asArray()[0]
      XCTAssertEqual(3 + 2, outValue)

      let neg = try graph.neg(add)

      let sevenTensor = try TF.Tensor.Scalar(Int32(7))

      let outs2 = try graph.runner().feed(feed, tensor: sevenTensor).fetch(neg).run()
      let out2 = outs2[0]
      XCTAssertEqual(out2.type ?? TF.DataType.dtInvalid, TF.DataType.dtInt32)
      XCTAssertEqual(0, out2.dimensionCount)
      XCTAssertEqual(MemoryLayout<Int32>.size, out.bytesCount)
      let outValue2:Int32 = try out2.asArray()[0]
      XCTAssertEqual(-(7 + 2), outValue2)

    }catch {
      XCTFail("session express: \(error)")
    }
  }

  func testSession() {
    do {
      let graph = try TF.Graph()
      let feed = try graph.placeholder()
      let two = try graph.scalar(2)
      let add = try graph.add(left: feed, right: two)


      let s = try graph.newSession()

      let threeTensor = try TF.Tensor.Scalar(Int32(3))

      let outs = try s.run(
        inputs: [(input: feed.output(0), tensor:threeTensor)],
        outputs: [add.output(0)])
      XCTAssertFalse(outs.isEmpty)
      let out = outs[0]
      XCTAssertEqual(out.type ?? TF.DataType.dtInvalid, TF.DataType.dtInt32)
      XCTAssertEqual(0, out.dimensionCount)
      XCTAssertEqual(MemoryLayout<Int32>.size, out.bytesCount)
      let outValue:Int32 = try out.asArray()[0]
      XCTAssertEqual(3 + 2, outValue)

      let neg = try graph.neg(add)

      let sevenTensor = try TF.Tensor.Scalar(Int32(7))

      let outs2 = try s.run(inputs: [(input: feed.output(0), tensor: sevenTensor)], outputs: [neg.output(0)])
      let out2 = outs2[0]
      XCTAssertEqual(out2.type ?? TF.DataType.dtInvalid, TF.DataType.dtInt32)
      XCTAssertEqual(0, out2.dimensionCount)
      XCTAssertEqual(MemoryLayout<Int32>.size, out.bytesCount)
      let outValue2:Int32 = try out2.asArray()[0]
      XCTAssertEqual(-(7 + 2), outValue2)

    }catch {
      XCTFail("session: \(error)")
    }
  }

  func testImportGraphDef () {
    do {
      let graph0 = try TF.Graph()
      _ = try graph0.placeholder()
      let three = try graph0.scalar(3)
      _ = try graph0.neg(three)

      _ = try graph0.searchOperation(forName: "feed")
      _ = try graph0.searchOperation(forName: "scalar")
      _ = try graph0.searchOperation(forName: "neg")

      let graph_def = graph0.definition!

      let opts = try TF.GraphDefOptions()
      opts.set(prefix: "imported")

      let graph1 = try TF.Graph()
      try graph1.import(definition: graph_def, options: opts)

      let feed = try graph1.searchOperation(forName: "imported/feed")
      let scalar = try graph1.searchOperation(forName: "imported/scalar")
      _ = try graph1.searchOperation(forName: "imported/neg")

      let opts2 = try TF.GraphDefOptions()
      opts2.set(prefix: "imported2")
      opts2.addInputMapping(sourceName: "scalar", sourceIndex: 0, destination: scalar.output(0))

      /** UNRESOLVED BUG HERE : unable to add return output `feed` **/
      //opts2.addReturnOutput(operationName: "feed", index: 0)
      opts2.addReturnOutput(operationName: "scalar", index: 0)

      let return_outputs = try graph1.importWithReturnOutputs(definition: graph_def, options: opts2)

      /* UNRESOLVED BUG HERE: unable to add return output - original script is 2 */
      XCTAssertEqual(1, return_outputs.count)

      let feed2 = try graph1.searchOperation(forName: "imported2/feed")
      _ = try graph1.searchOperation(forName: "imported2/scalar")
      _ = try graph1.searchOperation(forName: "imported2/neg")

      let opts3 = try TF.GraphDefOptions()
      opts3.set(prefix: "imported3")
      opts3.addControlDependency(operation: feed)
      opts3.addControlDependency(operation: feed2)
      try graph1.import(definition: graph_def, options: opts3)

      let feed3 = try graph1.searchOperation(forName: "imported3/feed")
      let scalar3 = try graph1.searchOperation(forName: "imported3/scalar")
      _ = try graph1.searchOperation(forName: "imported3/neg")

      let control_inputs = scalar3.controlInputs
      XCTAssertEqual(2, control_inputs.count)
      XCTAssertEqual(feed, control_inputs[0])
      XCTAssertEqual(feed2, control_inputs[1])

      let control_inputs2 = feed3.controlInputs
      XCTAssertEqual(2, control_inputs2.count)
      XCTAssertEqual(feed, control_inputs2[0])
      XCTAssertEqual(feed2, control_inputs2[1])

      let graph_def2 = graph1.definition!

      let opts4 = try TF.GraphDefOptions()
      opts4.set(prefix: "imported4")

      /** TESTING NOTE: THERE MAY BE A BUG OF REMAP CONTROL DEPENDENCY**/
      opts4.remapControlDependency(source: "imported/feed", destination: feed)
      try graph1.import(definition: graph_def2, options: opts4)

      let scalar4 = try graph1.searchOperation(forName: "imported4/imported3/scalar")
      let feed4 = try graph1.searchOperation(forName: "imported4/imported2/feed")

      let control_inputs3 = scalar4.controlInputs
      XCTAssertEqual(2, control_inputs3.count)
      XCTAssertEqual(feed4, control_inputs3[1])

      let feedX = control_inputs3[0]
      XCTAssertEqual(feedX.name ?? "", "imported4/imported/feed")
      XCTAssertEqual(feedX.type ?? "", "Placeholder")
      XCTAssertEqual(try feedX.attribute(forKey: "dtype").type, TF.DataType.dtInt32)

      let _ = try graph1.add(left: feed, right: scalar)
    }catch {
      XCTFail("import graph def: \(error)")
    }

  }

  func testGraph2() {
    do {
      let graph = try TF.Graph()
      let feed = try graph.placeholder()
      XCTAssertEqual(feed.name ?? "", "feed")
      XCTAssertEqual(feed.type ?? "", "Placeholder")
      XCTAssertTrue(feed.device.isEmpty)
      XCTAssertEqual(feed.numberOfOutputs, 1)
      let feedOut0 = feed.output(0)
      guard let tp = TF.Operation.TypeOf(output: feedOut0) else {
        XCTFail("graph: cannot get output type")
        return
      }//end guard
      XCTAssertEqual(tp, TF.DataType.dtInt32)
      let sz = try feed.sizeOfOutputList(argument: "output")
      XCTAssertEqual(sz, 1)
      XCTAssertEqual(0, feed.numberOfInputs)
      let consumers = TF.Operation.Consumers(output: feedOut0)
      XCTAssertEqual(0, consumers.count)
      XCTAssertEqual(0, feed.controlInputs.count)
      XCTAssertEqual(0, feed.controlOutputs.count)

      let attr_value = try feed.attribute(forKey: "dtype")
      XCTAssertEqual(attr_value.type, TF.DataType.dtInt32)

      let three = try graph.scalar(3)
      let add = try graph.add(left: feed, right: three)

      XCTAssertEqual("add", add.name ?? "")
      XCTAssertEqual("AddN", add.type ?? "")
      XCTAssertTrue(add.device.isEmpty)
      XCTAssertEqual(1, add.numberOfOutputs)
      XCTAssertEqual(TF.Operation.TypeOf(output: add.output(0)) ?? TF.DataType.dtInvalid, TF.DataType.dtInt32)

      XCTAssertEqual(1, try add.sizeOfOutputList(argument: "sum"))
      XCTAssertEqual(2, add.numberOfInputs)
      XCTAssertEqual(2, try add.sizeOfInputList(argument: "inputs"))
      let add0 = add.asInput(0)
      let add1 = add.asInput(1)
      XCTAssertEqual(TF.Operation.TypeOf(input: add0) ?? TF.DataType.dtInvalid, TF.DataType.dtInt32)
      XCTAssertEqual(TF.Operation.TypeOf(input: add1) ?? TF.DataType.dtInvalid, TF.DataType.dtInt32)
      let add_in_0 = TF.Operation.AsInput(input: add0)
      XCTAssertEqual(feed.operation, add_in_0.oper)
      XCTAssertEqual(0, add_in_0.index)
      let add_in_1 = TF.Operation.AsInput(input: add1)
      XCTAssertEqual(three.operation, add_in_1.oper)
      XCTAssertEqual(0, add_in_1.index)


      let consumption = TF.Operation.Consumers(output: add.output(0))
      XCTAssertTrue(consumption.isEmpty)
      XCTAssertTrue(add.controlInputs.isEmpty)
      XCTAssertTrue(add.controlOutputs.isEmpty)
      let valueT = try add.attribute(forKey: "T")
      let valueN = try add.attribute(forKey: "N")

      XCTAssertEqual(valueT.type, TF.DataType.dtInt32)
      XCTAssertEqual(valueN.i, 2)

      let three_port_array = TF.Operation.Consumers(output: three.output(0))
      XCTAssertEqual(1, three_port_array.count)
      let three_port = three_port_array[0]
      XCTAssertEqual(add.operation, three_port.oper)
      XCTAssertEqual(1, three_port.index)

      var def = graph.definition!

      let found_placeholders = def.node
        .filter { $0.name == "feed" && $0.op == "Placeholder" }
        .filter { n in
          if n.attr["dtype"]?.type == TF.DataType.dtInt32, let _ = n.attr["shape"] {
            return true
          }else {
            return false
          }//end if
        }//end filter
      XCTAssertFalse(found_placeholders.isEmpty)

      let found_scalarconst = def.node.filter { $0.op == "Const" && $0.name == "scalar" }
      XCTAssertFalse(found_scalarconst.isEmpty)

      let found_addN = def.node.filter { $0.op == "AddN" && $0.name == "add" && $0.input.count == 2 }
      XCTAssertFalse(found_addN.isEmpty)

      let neg = try graph.opBuilder(name: "neg", type: "Neg").add(input: add.output(0)).build()
      guard
        let node_def = neg.nodeDefinition,
        node_def.op == "Neg", node_def.name == "neg"
      else {
        XCTFail("graph2: getting node definition failed")
        return
      }//end guard
      XCTAssertEqual(1, node_def.input.count)
      XCTAssertEqual("add", node_def.input[0])

      def.node.append(node_def)
      guard let def2 = graph.definition else {
        XCTFail("graph2: second def fault")
        return
      }//end guard

      XCTAssertTrue(def2._protobuf_generated_isEqualTo(other: def))

      let neg2 = try graph.searchOperation(forName: "neg")
      XCTAssertEqual(neg, neg2)

      let node_def2 = neg2.nodeDefinition!
      XCTAssertTrue(node_def._protobuf_generated_isEqualTo(other: node_def2))

      let feed2 = try graph.searchOperation(forName: "feed")
      XCTAssertEqual(feed, feed2)

      XCTAssertTrue(feed.nodeDefinition!._protobuf_generated_isEqualTo(other: feed2.nodeDefinition!))

      let oplist = graph.operations

      XCTAssertTrue(oplist.contains(feed))
      XCTAssertTrue(oplist.contains(three))
      XCTAssertTrue(oplist.contains(add))
      XCTAssertTrue(oplist.contains(neg))
      XCTAssertTrue(oplist.filter {
        $0 != feed && $0 != three && $0 != add && $0 != neg
      }.isEmpty)

    } catch {
      XCTFail("graph2: \(error)")
    }
  }
  func testGraph() {
    do {
      let graph = try TF.Graph()
      let feed = try graph.placeholder()
      XCTAssertTrue(feed.device.isEmpty)
      XCTAssertEqual(feed.numberOfOutputs, 1)
      let feedOut0 = feed.output(0)
      guard let tp = TF.Operation.TypeOf(output: feedOut0) else {
        XCTFail("graph: cannot get output type")
        return
      }//end guard
      XCTAssertEqual(tp, TF.DataType.dtInt32)
      let sz = try feed.sizeOfOutputList(argument: "output")
      XCTAssertEqual(sz, 1)
      XCTAssertEqual(0, feed.numberOfInputs)
      let consumers = TF.Operation.Consumers(output: feedOut0)
      XCTAssertEqual(0, consumers.count)
      XCTAssertEqual(0, feed.controlInputs.count)
      XCTAssertEqual(0, feed.controlOutputs.count)
      let dtype = try feed.attribute(forKey: "dtype")
      XCTAssertEqual(dtype.type, TF.DataType.dtInt32)

      let three = try graph.scalar(3)
      let three_out_0 = three.output(0)
      let _ = try graph.getTensor(output: three_out_0)
    }catch TF.Panic.FAULT(let reason) {
      XCTAssertEqual(reason, "Shape has no dimensions")
    } catch {
      XCTFail("graph: \(error)")
    }
  }

  func testSetShapePlaceHolder() {
    do {
      let graph = try TF.Graph()
      let desc = try graph.opBuilder(name: "name", type: "Placeholder")
      .set(attributes: ["dtype": TF.DataType.dtInt32])
      let feed = try desc.build()
      let feed_out_0 = feed.output(0)
      let _ = try graph.getTensor(output: feed_out_0)
    }catch TF.Panic.FAULT(let reason) {
      XCTAssertEqual(reason, "Shape has no dimensions")
    }catch {
      XCTFail("SetShape: \(error)")
    }
  }

  func testSetShape() {
    do {
      let graph = try TF.Graph()
      let desc = try graph.opBuilder(name: "feed", type: "Placeholder")
      let feed = try desc.set(attributes: ["dtype": TF.DataType.dtInt32]).build()
      let feed_out_0 = feed.output(0)
      let dim:[Int64] = [2, -1]
      try graph.setTensor(output: feed_out_0, shape: dim)
      let dim2 = try graph.getTensor(output: feed_out_0)
      XCTAssertEqual(dim, dim2)
      let dim3: [Int64] = [2, 3]
      try graph.setTensor(output: feed_out_0, shape: dim3)
      let dim4 = try graph.getTensor(output: feed_out_0)
      XCTAssertEqual(dim3, dim4)
      let dim5: [Int64] = [-1, -1]
      try graph.setTensor(output: feed_out_0, shape: dim5)
      let dim6 = try graph.getTensor(output: feed_out_0)
      XCTAssertEqual(dim3, dim6)
    }catch {
      XCTFail("SetShape: \(error)")
    }
  }

  override func setUp() {
    do {
      #if os(Linux)
      try TF.Open(library:"/tmp/testdata/linux/lib/libtensorflow.so")
      #else
      try TF.Open(library:"/tmp/testdata/darwin/lib/libtensorflow.so")
      #endif
    }catch {
      XCTFail("\(error)")
    }
  }

  override func tearDown() {
    TF.Close()
  }

  func testVersion() {
    XCTAssertEqual(TF.Version, "1.3.0")
  }

  func testSize() {
    XCTAssertEqual(try TF.SizeOf(Type: .dtFloat), 4)
    XCTAssertEqual(try TF.SizeOf(Type: .dtDouble), 8)
    XCTAssertEqual(try TF.SizeOf(Type: .dtInt32), 4)
    XCTAssertEqual(try TF.SizeOf(Type: .dtUint8), 1)
    XCTAssertEqual(try TF.SizeOf(Type: .dtUint16), 2)
    XCTAssertEqual(try TF.SizeOf(Type: .dtInt8), 1)
    XCTAssertEqual(try TF.SizeOf(Type: .dtString), 0)
    XCTAssertEqual(try TF.SizeOf(Type: .dtComplex64), 8)
    XCTAssertEqual(try TF.SizeOf(Type: .dtInt64), 8)
    XCTAssertEqual(try TF.SizeOf(Type: .dtBool), 1)
    XCTAssertEqual(try TF.SizeOf(Type: .dtQint8), 1)
    XCTAssertEqual(try TF.SizeOf(Type: .dtQuint8), 1)
    XCTAssertEqual(try TF.SizeOf(Type: .dtQint32), 4)
    XCTAssertEqual(try TF.SizeOf(Type: .dtBfloat16), 0)
    XCTAssertEqual(try TF.SizeOf(Type: .dtQint8), 1)
    XCTAssertEqual(try TF.SizeOf(Type: .dtQuint16), 2)
    XCTAssertEqual(try TF.SizeOf(Type: .dtComplex128), 16)
    XCTAssertEqual(try TF.SizeOf(Type: .dtHalf), 2)
    XCTAssertEqual(try TF.SizeOf(Type: .dtResource), 0)
  }

  func testStatus () {
    do {
      let s0 = try TF.Status()
      XCTAssertEqual(s0.code!, TF.Code.OK)
    }catch {
      XCTFail("status: \(error)")
    }
  }

  func testBuffer () {
    do {
      let _ = try TF.Buffer()
      let b0 = try TF.Buffer(string: "hello")
      let b1 = try TF.Buffer(string: "world", size: 5)
      XCTAssertEqual(b0.valueString, "hello")
      XCTAssertEqual(b1.valueString, "world")
    }catch {
      XCTFail("buffer: \(error)")
    }
  }

  func testTensorScalarConst() {
    do {
      let x = try TF.Tensor.Scalar(100)
      let y: Int = try x.asScalar()
      XCTAssertEqual(100, y)
    }catch {
      XCTFail("tensor scalar const: \(error)")
    }
  }

  func testEncodeDecode() {
    do {
      let s = "Hello, world!"
      let s0 = s.utf8.map { Int8($0) }

      let encoded = try TF.Encode(string: s0)
      let s1 = try TF.Decode(string: encoded)
      XCTAssertEqual(s0, s1)

      let words = ["the", "quick", "brown", "fox", "jumped", "over"]
      let data = words.map { Data(bytes: $0.utf8.map { UInt8($0) } ) }

      let encoded2 = try TF.Encode(strings: data)
      let data2 = try TF.Decode(strings: encoded2, count: words.count)
      XCTAssertEqual(data, data2)
    }catch {
      XCTFail("encode decode: \(error)")
    }
  }

  func testSessionOptions() {
    do {
      let config = try TF.Config(jsonString: "{\"intra_op_parallelism_threads\": 4}")
      let _ = try TF.SessionOptions()
        .set(target: "local,127.0.0.1:8080,perfect.org:8181")
        .set(config: config)
    }catch {
      XCTFail("session options: \(error)")
    }
  }

  func testOpList() {
    do {
      let oplist = try TF.OperationList()
      XCTAssertGreaterThan(oplist.operations.count, 0)
    }catch {
      XCTFail("OpList: \(error)")
    }
  }
}
