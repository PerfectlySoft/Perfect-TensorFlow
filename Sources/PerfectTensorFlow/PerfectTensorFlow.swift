//
//  PerfectTensorFlow.swift
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

import Foundation
import TensorFlowAPI

public extension Array {

  /// method that can 'completely' flatten a multi-dimensional array
  public static func Flat (_ array: Array<Any>) -> Array<Any> {
    if let a = array as? Array<Array<Any>> {
      return Flat(a.flatMap({$0}))
    }
    return array.flatMap{$0}
  }//end func

  /// instance method
  public func flat() -> Array<Any> {
    return Array.Flat(self)
  }

  public var shape: [Int] {
    var _shape = [Int]()
    var a = self as Array<Any>
    while a.count > 0 {
      _shape.append(a.count)
      if let b = a as? Array<Array<Any>> , let c = b.first {
        a = c
      } else {
        break
      }//end if
    }//end while
    return _shape
  }//end var

  public func column(index: Int) -> Array<Any> {
    var b = [Any]()
    let s = shape
    guard s.count > 1, index > -1, index < s[1],
      let a = self as? Array<Array<Any>> else {
        if index > -1 && index < self.count {
          b.append(self[index])
        }//end if
        return b
    }//end guard
    a.forEach { c in
      b.append(c[index])
    }//next
    return b
  }//end func
}

typealias SwiftArray<T> = Array<T>
public extension Data {
  public static func From(_ string: String) -> Data {
    return string.withCString { p -> Data in
      return Data(bytes: p, count: string.utf8.count)
    }//end return
  }//end from
  public var string: String {
    return self.withUnsafeBytes { (p: UnsafePointer<CChar>) -> String in
      var q = Array(UnsafeBufferPointer(start: p, count: self.count))
      q.append(0)
      return String(cString: q)
    }//end return
  }//end var
}//end extension

/// Static C API of TensorFlow
public class TensorFlow {

  public typealias Input = TF_Input
  public typealias Output = TF_Output
  public typealias Panic = TFLib.Panic
  public typealias Code = TFLib.Code
  public typealias Config = Tensorflow_ConfigProto
  public typealias DataType = Tensorflow_DataType
  public typealias AllocationDescription = Tensorflow_AllocationDescription
  public typealias AttrValue = Tensorflow_AttrValue
  public typealias NameAttrList = Tensorflow_NameAttrList
  public typealias BigQueryTablePartition = Tensorflow_BigQueryTablePartition
  public typealias CheckpointState = Tensorflow_CheckpointState
  public typealias JobDef = Tensorflow_JobDef
  public typealias ClusterDef = Tensorflow_ClusterDef
  public typealias GPUOptions = Tensorflow_GPUOptions
  public typealias OptimizerOptions = Tensorflow_OptimizerOptions
  public typealias GraphOptions = Tensorflow_GraphOptions
  public typealias ThreadPoolOptionProto = Tensorflow_ThreadPoolOptionProto
  public typealias RPCOptions = Tensorflow_RPCOptions
  public typealias ConfigProto = Tensorflow_ConfigProto
  public typealias RunOptions = Tensorflow_RunOptions
  public typealias RunMetadata = Tensorflow_RunMetadata
  public typealias ValuesDef = Tensorflow_ValuesDef
  public typealias CondContextDef = Tensorflow_CondContextDef
  public typealias WhileContextDef = Tensorflow_WhileContextDef
  public typealias CostGraphDef = Tensorflow_CostGraphDef
  public typealias CppShapeInferenceResult = Tensorflow_CppShapeInferenceResult
  public typealias CppShapeInferenceInputsNeeded = Tensorflow_CppShapeInferenceInputsNeeded
  public typealias DebugTensorWatch = Tensorflow_DebugTensorWatch
  public typealias DebugOptions = Tensorflow_DebugOptions
  public typealias EventReply = Tensorflow_EventReply
  public typealias DeviceLocality = Tensorflow_DeviceLocality
  public typealias DeviceAttributes = Tensorflow_DeviceAttributes
  public typealias DeviceProperties = Tensorflow_DeviceProperties
  public typealias Event = Tensorflow_Event
  public typealias LogMessage = Tensorflow_LogMessage
  public typealias SessionLog = Tensorflow_SessionLog
  public typealias TaggedRunMetadata = Tensorflow_TaggedRunMetadata
  public typealias Example = Tensorflow_Example
  public typealias SequenceExample = Tensorflow_SequenceExample
  public typealias VarLenFeatureProto = Tensorflow_VarLenFeatureProto
  public typealias FixedLenFeatureProto = Tensorflow_FixedLenFeatureProto
  public typealias FeatureConfiguration = Tensorflow_FeatureConfiguration
  public typealias ExampleParserConfiguration = Tensorflow_ExampleParserConfiguration
  public typealias ExampleWithExtras = Tensorflow_ExampleWithExtras
  public typealias BytesList = Tensorflow_BytesList
  public typealias FloatList = Tensorflow_FloatList
  public typealias Int64List = Tensorflow_Int64List
  public typealias Feature = Tensorflow_Feature
  public typealias Features = Tensorflow_Features
  public typealias FeatureList = Tensorflow_FeatureList
  public typealias FeatureLists = Tensorflow_FeatureLists
  public typealias FunctionDefLibrary = Tensorflow_FunctionDefLibrary
  public typealias FunctionDef = Tensorflow_FunctionDef
  public typealias GradientDef = Tensorflow_GradientDef
  public typealias GraphDef = Tensorflow_GraphDef
  public typealias GraphTransferInfo = Tensorflow_GraphTransferInfo
  public typealias HParamDef = Tensorflow_HParamDef
  public typealias KernelDef = Tensorflow_KernelDef
  public typealias BoostedTrees_Learner_TreeRegularizationConfig = Tensorflow_BoostedTrees_Learner_TreeRegularizationConfig
  public typealias BoostedTrees_Learner_TreeConstraintsConfig = Tensorflow_BoostedTrees_Learner_TreeConstraintsConfig
  public typealias BoostedTrees_Learner_LearningRateConfig = Tensorflow_BoostedTrees_Learner_LearningRateConfig
  public typealias BoostedTrees_Learner_LearningRateFixedConfig = Tensorflow_BoostedTrees_Learner_LearningRateFixedConfig
  public typealias BoostedTrees_Learner_LearningRateLineSearchConfig = Tensorflow_BoostedTrees_Learner_LearningRateLineSearchConfig
  public typealias BoostedTrees_Learner_AveragingConfig = Tensorflow_BoostedTrees_Learner_AveragingConfig
  public typealias BoostedTrees_Learner_LearningRateDropoutDrivenConfig = Tensorflow_BoostedTrees_Learner_LearningRateDropoutDrivenConfig
  public typealias BoostedTrees_Learner_LearnerConfig = Tensorflow_BoostedTrees_Learner_LearnerConfig
  public typealias MemoryLogStep = Tensorflow_MemoryLogStep
  public typealias MemoryLogTensorAllocation = Tensorflow_MemoryLogTensorAllocation
  public typealias MemoryLogTensorDeallocation = Tensorflow_MemoryLogTensorDeallocation
  public typealias MemoryLogTensorOutput = Tensorflow_MemoryLogTensorOutput
  public typealias MemoryLogRawAllocation = Tensorflow_MemoryLogRawAllocation
  public typealias MemoryLogRawDeallocation = Tensorflow_MemoryLogRawDeallocation
  public typealias Serving_Signatures = Tensorflow_Serving_Signatures
  public typealias Serving_TensorBinding = Tensorflow_Serving_TensorBinding
  public typealias Serving_AssetFile = Tensorflow_Serving_AssetFile
  public typealias Serving_Signature = Tensorflow_Serving_Signature
  public typealias Serving_RegressionSignature = Tensorflow_Serving_RegressionSignature
  public typealias Serving_ClassificationSignature = Tensorflow_Serving_ClassificationSignature
  public typealias Serving_GenericSignature = Tensorflow_Serving_GenericSignature
  public typealias CreateSessionRequest = Tensorflow_CreateSessionRequest
  public typealias CreateSessionResponse = Tensorflow_CreateSessionResponse
  public typealias ExtendSessionRequest = Tensorflow_ExtendSessionRequest
  public typealias ExtendSessionResponse = Tensorflow_ExtendSessionResponse
  public typealias RunStepRequest = Tensorflow_RunStepRequest
  public typealias RunStepResponse = Tensorflow_RunStepResponse
  public typealias PartialRunSetupRequest = Tensorflow_PartialRunSetupRequest
  public typealias PartialRunSetupResponse = Tensorflow_PartialRunSetupResponse
  public typealias CloseSessionRequest = Tensorflow_CloseSessionRequest
  public typealias CloseSessionResponse = Tensorflow_CloseSessionResponse
  public typealias ResetRequest = Tensorflow_ResetRequest
  public typealias ResetResponse = Tensorflow_ResetResponse
  public typealias ListDevicesRequest = Tensorflow_ListDevicesRequest
  public typealias ListDevicesResponse = Tensorflow_ListDevicesResponse
  public typealias MemmappedFileSystemDirectoryElement = Tensorflow_MemmappedFileSystemDirectoryElement
  public typealias MemmappedFileSystemDirectory = Tensorflow_MemmappedFileSystemDirectory
  public typealias MetaGraphDef = Tensorflow_MetaGraphDef
  public typealias CollectionDef = Tensorflow_CollectionDef
  public typealias TensorInfo = Tensorflow_TensorInfo
  public typealias SignatureDef = Tensorflow_SignatureDef
  public typealias AssetFileDef = Tensorflow_AssetFileDef
  public typealias NamedTensorProto = Tensorflow_NamedTensorProto
  public typealias NodeDef = Tensorflow_NodeDef
  public typealias OpDef = Tensorflow_OpDef
  public typealias OpDeprecation = Tensorflow_OpDeprecation
  public typealias OpList = Tensorflow_OpList
  public typealias OpGenOverride = Tensorflow_OpGenOverride
  public typealias OpGenOverrides = Tensorflow_OpGenOverrides
  public typealias OpInfo = Tensorflow_OpInfo
  public typealias OpPerformance = Tensorflow_OpPerformance
  public typealias OpPerformanceList = Tensorflow_OpPerformanceList
  public typealias SpriteMetadata = Tensorflow_SpriteMetadata
  public typealias EmbeddingInfo = Tensorflow_EmbeddingInfo
  public typealias ProjectorConfig = Tensorflow_ProjectorConfig
  public typealias QueueRunnerDef = Tensorflow_QueueRunnerDef
  public typealias ReaderBaseState = Tensorflow_ReaderBaseState
  public typealias RemoteFusedGraphExecuteInfo = Tensorflow_RemoteFusedGraphExecuteInfo
  public typealias ResourceHandle = Tensorflow_ResourceHandleProto
  public typealias AutoParallelOptions = Tensorflow_AutoParallelOptions
  public typealias RewriterConfig = Tensorflow_RewriterConfig
  public typealias SavedModel = Tensorflow_SavedModel
  public typealias SavedSliceMeta = Tensorflow_SavedSliceMeta
  public typealias SavedTensorSliceMeta = Tensorflow_SavedTensorSliceMeta
  public typealias SavedSlice = Tensorflow_SavedSlice
  public typealias SavedTensorSlices = Tensorflow_SavedTensorSlices
  public typealias SaverDef = Tensorflow_SaverDef
  public typealias AllocatorMemoryUsed = Tensorflow_AllocatorMemoryUsed
  public typealias NodeOutput = Tensorflow_NodeOutput
  public typealias MemoryStats = Tensorflow_MemoryStats
  public typealias NodeExecStats = Tensorflow_NodeExecStats
  public typealias DeviceStepStats = Tensorflow_DeviceStepStats
  public typealias StepStats = Tensorflow_StepStats
  public typealias SummaryDescription = Tensorflow_SummaryDescription
  public typealias HistogramProto = Tensorflow_HistogramProto
  public typealias Summary = Tensorflow_Summary
  public typealias TensorProto = Tensorflow_TensorProto
  public typealias BundleHeaderProto = Tensorflow_BundleHeaderProto
  public typealias BundleEntryProto = Tensorflow_BundleEntryProto
  public typealias TensorDescription = Tensorflow_TensorDescription
  public typealias TensorShapeProto = Tensorflow_TensorShapeProto
  public typealias TensorSliceProto = Tensorflow_TensorSliceProto
  public typealias ServerDef = Tensorflow_ServerDef
  public typealias Test_TestAllTypes = Tensorflow_Test_TestAllTypes
  public typealias Test_NestedTestAllTypes = Tensorflow_Test_NestedTestAllTypes
  public typealias Test_ForeignMessage = Tensorflow_Test_ForeignMessage
  public typealias Test_TestEmptyMessage = Tensorflow_Test_TestEmptyMessage
  public typealias EntryValue = Tensorflow_EntryValue
  public typealias BenchmarkEntry = Tensorflow_BenchmarkEntry
  public typealias BenchmarkEntries = Tensorflow_BenchmarkEntries
  public typealias BuildConfiguration = Tensorflow_BuildConfiguration
  public typealias CommitId = Tensorflow_CommitId
  public typealias CPUInfo = Tensorflow_CPUInfo
  public typealias MemoryInfo = Tensorflow_MemoryInfo
  public typealias GPUInfo = Tensorflow_GPUInfo
  public typealias PlatformInfo = Tensorflow_PlatformInfo
  public typealias AvailableDeviceInfo = Tensorflow_AvailableDeviceInfo
  public typealias MachineConfiguration = Tensorflow_MachineConfiguration
  public typealias RunConfiguration = Tensorflow_RunConfiguration
  public typealias TestResults = Tensorflow_TestResults
  public typealias Tfcompile_TensorId = Tensorflow_Tfcompile_TensorId
  public typealias Tfcompile_Feed = Tensorflow_Tfcompile_Feed
  public typealias Tfcompile_Fetch = Tensorflow_Tfcompile_Fetch
  public typealias Tfcompile_Config = Tensorflow_Tfcompile_Config
  public typealias Tfprof_CodeDef = Tensorflow_Tfprof_CodeDef
  public typealias Tfprof_OpLogEntry = Tensorflow_Tfprof_OpLogEntry
  public typealias Tfprof_OpLog = Tensorflow_Tfprof_OpLogProto
  public typealias Tfprof_OptionsProto = Tensorflow_Tfprof_OptionsProto
  public typealias Tfprof_TFProfTensorProto = Tensorflow_Tfprof_TFProfTensorProto
  //public typealias Tfprof_TFGraphNodeProto = Tensorflow_Tfprof_TFGraphNodeProto
  //public typealias Tfprof_TFCodeNodeProto = Tensorflow_Tfprof_TFCodeNodeProto
  public typealias Contrib_Tensorboard_TraceInfo = Tensorflow_Contrib_Tensorboard_TraceInfo
  public typealias Contrib_Tensorboard_OpInfo = Tensorflow_Contrib_Tensorboard_OpInfo
  public typealias Contrib_Tensorboard_LineTrace = Tensorflow_Contrib_Tensorboard_LineTrace
  public typealias Contrib_Tensorboard_TensorInfo = Tensorflow_Contrib_Tensorboard_TensorInfo
  public typealias Contrib_Tensorboard_FileInfo = Tensorflow_Contrib_Tensorboard_FileInfo
  public typealias BoostedTrees_Trees_TreeNode = Tensorflow_BoostedTrees_Trees_TreeNode
  public typealias BoostedTrees_Trees_TreeNodeMetadata = Tensorflow_BoostedTrees_Trees_TreeNodeMetadata
  public typealias BoostedTrees_Trees_Leaf = Tensorflow_BoostedTrees_Trees_Leaf
  public typealias BoostedTrees_Trees_Vector = Tensorflow_BoostedTrees_Trees_Vector
  public typealias BoostedTrees_Trees_SparseVector = Tensorflow_BoostedTrees_Trees_SparseVector
  public typealias BoostedTrees_Trees_DenseFloatBinarySplit = Tensorflow_BoostedTrees_Trees_DenseFloatBinarySplit
  public typealias BoostedTrees_Trees_SparseFloatBinarySplitDefaultLeft = Tensorflow_BoostedTrees_Trees_SparseFloatBinarySplitDefaultLeft
  public typealias BoostedTrees_Trees_SparseFloatBinarySplitDefaultRight = Tensorflow_BoostedTrees_Trees_SparseFloatBinarySplitDefaultRight
  public typealias BoostedTrees_Trees_CategoricalIdBinarySplit = Tensorflow_BoostedTrees_Trees_CategoricalIdBinarySplit
  public typealias BoostedTrees_Trees_CategoricalIdSetMembershipBinarySplit = Tensorflow_BoostedTrees_Trees_CategoricalIdSetMembershipBinarySplit
  public typealias BoostedTrees_Trees_DecisionTreeConfig = Tensorflow_BoostedTrees_Trees_DecisionTreeConfig
  public typealias BoostedTrees_Trees_DecisionTreeMetadata = Tensorflow_BoostedTrees_Trees_DecisionTreeMetadata
  public typealias BoostedTrees_Trees_GrowingMetadata = Tensorflow_BoostedTrees_Trees_GrowingMetadata
  public typealias BoostedTrees_Trees_DecisionTreeEnsembleConfig = Tensorflow_BoostedTrees_Trees_DecisionTreeEnsembleConfig
  public typealias VariableDef = Tensorflow_VariableDef
  public typealias SaveSliceInfoDef = Tensorflow_SaveSliceInfoDef
  public typealias Channel = Tensorflow_Channel
  public typealias MemoryRegion = Tensorflow_MemoryRegion
  public typealias GetRemoteAddressRequest = Tensorflow_GetRemoteAddressRequest
  public typealias GetRemoteAddressResponse = Tensorflow_GetRemoteAddressResponse
  public typealias VersionDef = Tensorflow_VersionDef
  public typealias GetStatusRequest = Tensorflow_GetStatusRequest
  public typealias GetStatusResponse = Tensorflow_GetStatusResponse
  public typealias CreateWorkerSessionRequest = Tensorflow_CreateWorkerSessionRequest
  public typealias CreateWorkerSessionResponse = Tensorflow_CreateWorkerSessionResponse
  public typealias RegisterGraphRequest = Tensorflow_RegisterGraphRequest
  public typealias RegisterGraphResponse = Tensorflow_RegisterGraphResponse
  public typealias DeregisterGraphRequest = Tensorflow_DeregisterGraphRequest
  public typealias DeregisterGraphResponse = Tensorflow_DeregisterGraphResponse
  public typealias CleanupAllRequest = Tensorflow_CleanupAllRequest
  public typealias CleanupAllResponse = Tensorflow_CleanupAllResponse
  public typealias ExecutorOpts = Tensorflow_ExecutorOpts
  public typealias RunGraphRequest = Tensorflow_RunGraphRequest
  public typealias RunGraphResponse = Tensorflow_RunGraphResponse
  public typealias CleanupGraphRequest = Tensorflow_CleanupGraphRequest
  public typealias CleanupGraphResponse = Tensorflow_CleanupGraphResponse
  public typealias RecvTensorRequest = Tensorflow_RecvTensorRequest
  public typealias RecvTensorResponse = Tensorflow_RecvTensorResponse
  public typealias LoggingRequest = Tensorflow_LoggingRequest
  public typealias LabeledStepStats = Tensorflow_LabeledStepStats
  public typealias LoggingResponse = Tensorflow_LoggingResponse
  public typealias TraceOpts = Tensorflow_TraceOpts
  public typealias TracingRequest = Tensorflow_TracingRequest
  public typealias TracingResponse = Tensorflow_TracingResponse

  /// DLL Loader, must be called before all methods
  /// - parameters:
  ///   - library: path of libtensorflow to load
  /// - throws: Panic
  public static func Open (library: String = "/usr/local/lib/libtensorflow.so") throws {
    try TFLib.Open(library)
    let r = TF_LoadPatchLibrary(library)
    guard r == 0 else { throw Panic.DLL(reason: "Unable to Load Patch")}
  }//end func

  /// DLL resource release
  public static func Close() {
    if let _ = TFLib.libDLL {
      TFLib.Close()
    }//end if
    // TF_ClosePatchLibrary()
  }//end func

  /// Express Class Wrapper of TF_Status
  public class Status {
    let status : OpaquePointer

    /// create a blank status object
    public init() throws {
      guard let _ = TFLib.libDLL, let s = TFLib.NewStatus() else {
        throw Panic.CALL
      }//end guard
      status = s
    }//end if

    deinit {
      TFLib.DeleteStatus(status)
    }//end deinit

    /// get the code from the status
    public var code: Code? {
      get {
        return TFLib.Code(rawValue: Int(TFLib.GetCode(status)))
      }
    }

    /// get the message from the status
    public var message: String {
      get {
        if let m = TFLib.Message(status) {
          return String(cString: m)
        } else {
          return ""
        }//end if
      }//end get
    }//end message
  }//end status

  /// Express Class Wrapper of TF_Buffer
  public class Buffer : CustomStringConvertible {

    var buffer: UnsafeMutablePointer<TF_Buffer>
    var autoDestroy = true

    /// create an empty buffer
    public init() throws {
      guard let _ = TFLib.libDLL, let buf = TFLib.NewBuffer()
        else { throw Panic.CALL }
      buffer = buf
    }//end init

    /// create a buffer from a string
    /// - parameters:
    ///   - string: the string source
    ///   - size: optional. only set if the sze is not string length
    public init(string: String, size: Int = 0) throws {
      guard let _ = TFLib.libDLL,
        let buf = TFLib.NewBufferFromString(
          string, size < 1 ? string.utf8.count : size)
        else { throw Panic.CALL }
      buffer = buf
    }//end init

    /// create a buffer by duplicate its handle
    /// - parameters:
    ///   - buf: the buffer pointer handler to copy with
    public init(buf: UnsafeMutablePointer<TF_Buffer>) {
      buffer = buf
      autoDestroy = false
    }//end init

    /// create a buffer from data
    /// - parameters:
    /// - data: data to copy with
    public init(data: Data) throws {
      let pData = data.withUnsafeBytes {
        (ptr: UnsafePointer<Int8>) -> UnsafePointer<Int8> in
        return ptr
      }//end let
      guard let _ = TFLib.libDLL,
      let buf = TFLib.NewBufferFromString(pData, data.count)
      else { throw Panic.CALL }
      buffer = buf
    }//end public

    deinit {
      if autoDestroy {
        TFLib.DeleteBuffer(buffer)
      }//end if
    }//end deinit

    /// get data
    public var data: Data? {
      get {
        let b = buffer.pointee
        guard b.length > 0 else { return nil }
        return Data(bytes: b.data, count: b.length)
      }//end get
    }//end data

    /// get the data as a string
    public var valueString: String {
      get {
        let b = buffer.pointee
        guard b.length > 0 else { return "" }
        return String(cString: b.data.bindMemory(to: CChar.self, capacity: b.length))
      }//end get
    }//end var

    public var description: String {
      get {
        if autoDestroy {
          return "(auto): \(buffer.pointee)"
        } else {
          return "(manual): \(buffer.pointee)"
        }//end if
      }//end get
    }//end var
  }//end class

  /// utility for convert a Swift type to a Tensorflow type
  public final class DType<T> : CustomStringConvertible {

    /// get tensorflow type from the Swift one, will be nil if not available
    public var matched: DataType? {
      get {
        switch "\(T.self)" {
        case "Float":  return  DataType.dtFloat
        case "Double":  return  DataType.dtDouble
        case "Int32":  return  DataType.dtInt32
        case "UInt8":  return  DataType.dtUint8
        case "Int16":  return  DataType.dtInt16
        case "Int8":  return  DataType.dtInt8
        case "Int64", "Int":  return  DataType.dtInt64
        case "Bool":  return  DataType.dtBool
        case "UInt16":  return  DataType.dtUint16
        case "String", "Data": return DataType.dtString
        default:
          return nil
        }//end switch
      }//end case
    }//end var

    /// size of the type
    public var size: Int? {
      get {
        guard let tp = matched else { return nil }
        do {
          return try TensorFlow.SizeOf(Type: tp)
        }catch {
          return nil
        }
      }//end get
    }//end var

    public var description: String {
      get {
        return "<\(T.self): \(size ?? 0)>"
      }//end get
    }//end var
  }//end class DType

  /// DataTypeSize returns the sizeof() for the underlying type corresponding
  /// to the given TF_DataType enum value. Returns 0 for variable length types
  /// (eg. TF_STRING) or on failure.
  /// - parameters:
  ///   - Type: the data type to evaluate
  public static func SizeOf(Type: DataType) throws -> Int {
    guard let _ = TFLib.libDLL else { throw Panic.CALL }
    return Int(TFLib.DataTypeSize(Int32(Type.rawValue)))
  }//end SizeOf

  /// Express wrapper of Tensor
  public class Tensor : CustomStringConvertible, Equatable {
    let tensor : OpaquePointer
    internal var autoDestroy = true

    /// compare two operations
    public static func == (a: Tensor, b: Tensor) -> Bool {
      return a.data == b.data && a.dim == b.dim
    }//end func

    /// get a buffer copy from the tensor value
    public var data: [Int8] {
      get {
        let sz = self.bytesCount
        guard sz > 0, let p = TFLib.TensorData(tensor) else { return [Int8]() }
        let q = p.bindMemory(to: Int8.self, capacity: sz)
        let buffered = UnsafeBufferPointer(start: q, count: sz)
        return SwiftArray(buffered)
      }//end get
      set {
        let sz = self.bytesCount
        guard sz == newValue.count,
          let p = TFLib.TensorData(tensor) else { return }
        newValue.withUnsafeBufferPointer { pBuf in
          if let q = pBuf.baseAddress {
            p.copyBytes(from: q, count: newValue.count)
          }//end if
        }//end with
      }//end set
    }//end data

    /// Generate a tensor from a handler pointer.
    /// **NOTE** tensors created by this constructor will not deallocate automatically.
    /// - parameters:
    ///   - handle: the tensor handle pointer to copy with
    public init(handle: OpaquePointer) {
      tensor = handle
      autoDestroy = false
    }//end init

    /// Generate an empty tensor; User must set the data value after creation.
    /// Please **DO NOT** call it directly, use Tensor.Scalar or Tensor.Array instead.
    /// - parameters:
    ///   - dataType: type of the constant
    ///   - size: size of the memory to allocate. Zero for auto computation.
    ///   - dimensions: dimensions of the tensor. empty for scalar
    /// - throws: Panic
    public init(dataType: DataType, size: Int, dimensions: [Int64] = []) throws {

      guard let _ = TFLib.libDLL, size > 0
        else { throw Panic.INVALID }

      let tp = Int32(dataType.rawValue)

      if dimensions.count < 1 {
        guard let t = TFLib.AllocateTensor (tp, nil, 0, size) else {
          throw Panic.INVALID
        }//end guard

        tensor = t
        return
      }//end if

      tensor = try dimensions.withUnsafeBufferPointer { dimbuf -> OpaquePointer in
        guard let pDim = dimbuf.baseAddress,
        let t = TFLib.AllocateTensor(tp, pDim, Int32(dimensions.count), size)
        else { throw Panic.INVALID }
        return t
      }//end tensor
    }//end public

    /// Create a tensor by passing its value
    /// type can be Int, Float,... or String / Data
    /// - parameters:
    ///   - value: value of the tensor to pass
    /// - returns:
    ///   Tensor
    /// - throws:
    ///   Panic
    public static func Scalar<T> (_ value: T) throws -> Tensor {
      guard let _ = TFLib.libDLL, let tp = DType<T>().matched else {
        throw Panic.INVALID
      }//end guard
      let sz = try TensorFlow.SizeOf(Type: tp)
      if sz > 0 {
        let t = try Tensor(dataType: tp, size: sz)
        try t.withDataPointer { pData in
          let p = pData.bindMemory(to: T.self, capacity: 1)
          p.pointee = value
        }//end try
        return t
      }//end if
      if value is String, let v = value as? String {
        let data = try TensorFlow.Encode(strings: [ Data.From(v) ])
        let t = try Tensor(dataType: tp, size: data.count)
        t.data = data
        return t
      } else if value is Data, let v = value as? Data {
        let data = try TensorFlow.Encode(strings: [ v ])
        let t = try Tensor(dataType: tp, size: data.count)
        t.data = data
        return t
      }else {
        throw Panic.INVALID
      }//end if
    }//end tensor

    /// Create a tensor from a Matrix. *NOTE* Element must be number
    /// - parameters:
    ///   - matrix: Matrix in form of Array[Array[Array ... Array[Number]]]
    ///   - throws: Panic.FAULT
    /// - returns: tensor
    public static func Matrix(_ matrix: Array<Any>) throws -> Tensor {
      let shape = matrix.shape.map { Int64($0) }
      let flattened = matrix.flat()
      let f: [Float]
      if flattened is [UInt8], let i = flattened as? [UInt8] {
        f = i.map { Float($0) }
      } else if flattened is [UInt32], let i = flattened as? [UInt32] {
        f = i.map { Float($0) }
      } else if flattened is [UInt], let i = flattened as? [UInt] {
        f = i.map { Float($0) }
      } else if flattened is [UInt64], let i = flattened as? [UInt64] {
        f = i.map { Float($0) }
      } else if flattened is [Int8], let i = flattened as? [Int8] {
        f = i.map { Float($0) }
      } else if flattened is [Int32], let i = flattened as? [Int32] {
        f = i.map { Float($0) }
      } else if flattened is [Int], let i = flattened as? [Int] {
        f = i.map { Float($0) }
      } else if flattened is [Int64], let i = flattened as? [Int64] {
        f = i.map { Float($0) }
      } else if flattened is [Float], let i = flattened as? [Float] {
        f = i
      } else if flattened is [Double], let i = flattened as? [Double] {
        f = i.map { Float($0) }
      } else {
        throw Panic.FAULT(reason: "Matrix Element Must Be Number")
      }//end if
      return try Array(dimensions: shape, value: f)
    }

    /// Create a tensor by passing its value
    /// type can be Int, Float,... or String / Data
    /// - parameters:
    ///   - dimensions: the shape / dimensions of the tensor. for example, python tf.constant([[1,2,3],[4,5,6]]) will be dimensions: [2,3] here
    ///   - value: flattened value array of the tensor to pass, i.e., no matter how many dimensions the tensor has, the value must be flattened into a serialized array.  e.g, python tf.constant([[1,2,3],[4,5,6]]) will be equivalent to try Tensor.Array(dimensions: [2,3], [1,2,3,4,5,6])
    /// - returns:
    ///   Tensor
    /// - throws:
    ///   Panic

    public static func Array<T> (dimensions:[Int64], value: [T]) throws -> Tensor {
      let count = Int(dimensions.reduce(1) { $0 * $1 })
      guard let _ = TFLib.libDLL,
        let tp = DType<T>().matched, dimensions.count > 0,
        value.count == count else {
        throw Panic.INVALID
      }//end guard
      let unitSize = try TensorFlow.SizeOf(Type: tp)
      if unitSize > 0 {
        let size = unitSize * count
        let t = try Tensor(dataType: tp, size: size, dimensions: dimensions)
        guard let p = TFLib.TensorData(t.tensor) else {
          throw Panic.CALL
        }//end guard
        try value.withUnsafeBufferPointer { pValue in
          guard let q = pValue.baseAddress else { throw Panic.INVALID }
          p.copyBytes(from: q, count: size)
        }//end try
        return t
      }//end size

      let dataArray: [Data]

      if value is [String], let v = value as? [String] {
        dataArray = v.map { Data.From($0) }
      } else if value is [Data], let v = value as? [Data] {
        dataArray = v
      } else {
        throw Panic.INVALID
      }//end if

      let buf = try TensorFlow.Encode(strings: dataArray)
      let t = try Tensor(dataType: tp, size: buf.count, dimensions: dimensions)
      t.data = buf
      return t
    }

    deinit {
      if autoDestroy {
        TFLib.DeleteTensor(tensor)
      }//end if
    }

    /// check data type of the value / element of value array
    public var `type`: DataType? {
      get {
        return DataType(rawValue: Int(TFLib.TensorType(tensor)))
      }
    }

    /// check dimension count
    public var dimensionCount: Int {
      get {
        return Int(TFLib.NumDims(tensor))
      }
    }

    /// get a dimension data giving a proper index
    public func dimension(_ index: Int) throws -> Int64 {
      guard index >= 0, index < dimensionCount else { throw Panic.INVALID }
      return TFLib.Dim(tensor, Int32(index))
    }

    /// get total size of memory in bytes
    public var bytesCount: Int {
      get {
        return Int(TFLib.TensorByteSize(tensor))
      }
    }

    /// dimensions
    public var dim: [Int64] {
      get {
        do {
          let sz = dimensionCount
          var array = [Int64]()
          if sz > 0 {
            for i in 0 ... sz - 1 {
              array.append(try dimension(i))
            }//next i
          }//end if
          return array
        }catch {
          return [Int64]()
        }
      }
    }

    /// perform data pointer operations unsafely.
    public func withDataPointer<R>(body: (UnsafeMutableRawPointer) throws -> R) throws -> R {
      guard let data = TFLib.TensorData(tensor) else { throw Panic.CALL }
      return try body(data)
    }//end

    /// retrieve the data in form of an array
    public func asArray<T>() throws -> [T] {
      guard let data = TFLib.TensorData(tensor) else { throw Panic.CALL }
      let size = bytesCount
      let count = size / MemoryLayout<T>.size

      guard self.type != DataType.dtString && size > 0 && count > 0 else {
        throw Panic.INVALID
      }//end if

      let p = data.bindMemory(to: T.self, capacity: count)
      let array = UnsafeBufferPointer<T>(start: p, count: count)
      return SwiftArray(array)
    }//end func asArray

    public var strings: [Data] {
      let count = Int( self.dim.reduce(1) {$0 * $1} )
      guard self.type == DataType.dtString && count > 0 else { return [Data]() }
      do {
        return try TensorFlow.Decode(strings: self.data, count: count)
      }catch {
        return [Data]()
      }//end do
    }//end strings

    public func asScalar<T> () throws -> T {
      guard let data = TFLib.TensorData(tensor) else { throw Panic.CALL }
      let size = bytesCount
      guard size == MemoryLayout<T>.size else { throw Panic.CALL }
      let p = data.bindMemory(to: T.self, capacity: 1)
      return p.pointee
    }//end asScalar

    public var description: String {
      get {
        return "Tensor of \(self.type ?? .dtInvalid): <\(self.dimensionCount), \(self.bytesCount)>"
      }//end get
    }//end var
  }

  public static func Encode(strings: [Data]) throws -> [Int8] {

    var headers = [UInt64(0)]

    var body = [Int8]()

    // encoding / decoding is position sensitive, so must put it in order explicitly
    var size = UInt64(0)

    for i in 0 ... strings.count - 1 {
      // *NOTE* DON'T USE MAP
      // UInt8(128) to Int8 will cause segment fault

      let s = strings[i].withUnsafeBytes { (ptr: UnsafePointer<Int8>) -> [Int8] in
        let buffered = UnsafeBufferPointer(start: ptr, count: strings[i].count)
        return Array(buffered)
      }

      let encoded = try TensorFlow.Encode(string: s)
      size += UInt64(encoded.count)
      if i < (strings.count - 1) {
        headers.append(size)
      }//end if
      body.append(contentsOf: encoded)
    }//next i

    guard size == UInt64(body.count) else
    { throw Panic.FAULT(reason: "Coding Length Mismatched: \(size) -> \(body.count)") }

    var result = try headers.withUnsafeBufferPointer { pHeader -> [Int8] in
      guard let h = pHeader.baseAddress else { throw Panic.CALL }
      let sz = headers.count * MemoryLayout<UInt64>.size
      let buf = UnsafeRawPointer(h).bindMemory(to: Int8.self, capacity: sz)
      return Array(UnsafeBufferPointer(start: buf, count: sz))
    }//end try
    result.append(contentsOf: body)
    return result
  }//end encode

  public static func Decode(strings: [Int8], count: Int) throws -> [Data] {
    let szHeader = count * MemoryLayout<UInt64>.size
    guard count > 0, strings.count > 0, strings.count > szHeader else {
      throw Panic.INVALID
    }//end guard
    return try strings.withUnsafeBufferPointer { pHeader -> [Data] in
      guard let p = pHeader.baseAddress else { throw Panic.CALL }
      let buf = UnsafeRawPointer(p).bindMemory(to: UInt64.self, capacity: count)
      let header = Array( UnsafeBufferPointer(start: buf, count: count))
      var body = [Data]()
      let base = p.advanced(by: szHeader)
      for i in 0 ... header.count - 1 {
        let a = header[i]
        let len: Int
        if i < (header.count - 1) {
          len = Int(header[i + 1] - a)
        } else {
          len = strings.count - szHeader - Int(a)
        }//end if
        let buf = UnsafeBufferPointer(start: base.advanced(by: Int(a)), count: len)
        let array = Array(buf)
        let decoded = try TensorFlow.Decode(string: array)
        let string = try decoded.withUnsafeBufferPointer { pBuf -> Data in
          guard let p = pBuf.baseAddress else { throw Panic.CALL }
          return Data(bytes: p, count: decoded.count)
        }//end string
        body.append(string)
      }//next
      return body
    }//end header
  }//end func

  /// encode the source data
  /// - parameters:
  ///   - string: source data to encode
  /// - throws: Panic
  public static func Encode(string: [Int8]) throws -> [Int8] {

    return try string.withUnsafeBufferPointer{ pointer -> [Int8] in

      guard string.count > 0, let ptr = pointer.baseAddress
        else { throw Panic.INVALID }

      let sz = TFLib.StringEncodedSize(string.count)

      guard sz > 0 else { throw Panic.INVALID }

      let status = try Status()

      let dst = UnsafeMutablePointer<CChar>.allocate(capacity: sz)

      defer {
        dst.deallocate(capacity: sz)
      }//end defer

      let len = TFLib.StringEncode(ptr, string.count, dst, sz, status.status)
      guard sz == len, status.code == .OK else { throw Panic.CALL }

      return Array( UnsafeBufferPointer(start: dst, count: len) )
    }//end try
  }//func

  /// decode the objective data
  /// - parameters:
  ///   - string: data to decode
  /// - throws: Panic
  public static func Decode(string: [Int8]) throws -> [Int8] {

    return try string.withUnsafeBufferPointer{ pointer -> [Int8] in

      guard string.count > 0, let ptr = pointer.baseAddress
        else { throw Panic.INVALID }

      let status = try Status()

      var dst = UnsafeMutablePointer<CChar>(bitPattern: 0)

      var len = 0

      let sz = TFLib.StringDecode(ptr, string.count, &dst, &len, status.status)

      guard sz > 0, len > 0, status.code == .OK, let pd = dst else { throw Panic.CALL }

      return Array( UnsafeBufferPointer(start: pd, count: len ))
    }//end decode
  }//end func

  /// Options to configure a session
  public class SessionOptions {

    let options: OpaquePointer

    /// Generate an option object
    /// - throws: Panic
    public init () throws {
      guard let _ = TFLib.libDLL, let opt = TFLib.NewSessionOptions() else {
        throw Panic.CALL
      }
      options = opt
    }//end init

    deinit {
      TFLib.DeleteSessionOptions(options)
    }//end destructor

    /// Set the target in options
    /// - parameters:
    ///   - target: String which can be empty, a single entry, or a comma separated list of entries. Each entry is in one of the following formats : "local", "ip:port" or "host:port"
    public func `set`(target: String) -> SessionOptions {
      TFLib.SetTarget(options, target)
      return self
    }//end write

    /// Set the config in TF_SessionOptions.options. config should be a serialized tensorflow.ConfigProto proto. If config was not parsed successfully as a ConfigProto, record the error information in *status.
    /// - parameters:
    ///   - config: String which can be empty, a single entry, or a comma separated list of entries. Each entry is in one of the following formats : "local", "ip:port" or "host:port"
    /// throws: Panic
    public func `set`(config: Config) throws -> SessionOptions {
      let s = try Status()
      let data = try config.serializedData()
      let p = data.withUnsafeBytes { (ptr: UnsafePointer<CChar>) in return ptr }
      TFLib.SetConfig(options, p, data.count, s.status)
      guard s.code == .OK else { throw Panic.FAULT(reason: s.message) }
      return self
    }//end func
  }//end Session

  /// Tensor Shape Object
  public struct Shape {
    public var dimensions = [Int64]()
    public var protobuffer: TensorShapeProto? = nil
  }//end class

  /// Pre-object to build operations
  /// **NOTE** use graph.opBuilder() to create a builder and use add()/set() to chain it up.
  /// e.g. let op = try self.opBuilder(name, type).set(attributes: ["value": tensor, "dtype": tp]).build()
  public class OperationBuilder {

    /// handler pointer
    let descriptor: OpaquePointer

    /// - parameters:
    ///   - graph: graph to attach
    ///   - name: name of the operation
    ///   - type: type of the operation
    /// - throws: Panic
    public init(graph: Graph, name: String, `type`: String) throws {
      guard let _ = TFLib.libDLL else { throw Panic.DLL(reason: "Library is Missing") }
      guard let desc = TFLib.NewOperation(graph.graph, type, name) else {
        throw Panic.FAULT(reason: "NewOperation() failed")
      }//end guard
      descriptor = desc
    }//end init

    /// add an input to the operation
    /// - parameters:
    ///   - input: an input to add with
    /// - throws: Panic
    /// - returns: OperationBuilder object after setting
    public func add(input: Output) -> OperationBuilder {
      TFLib.AddInput(descriptor, input)
      return self
    }//end func

    /// add an input array to the operation
    /// - parameters:
    ///   - inputs: an array of input to add with
    /// - throws: Panic
    /// - returns: OperationBuilder object after setting
    public func add(inputs: [Output]) throws -> OperationBuilder {
      guard let p = (inputs.withUnsafeBufferPointer {
        ptr in return ptr.baseAddress })
        else {
        throw Panic.INVALID
      }//end guard
      TFLib.AddInputList(descriptor, p, Int32(inputs.count))
      return self
    }//end add

    /// add an operation to the builder
    /// - parameters:
    ///   - control: a control operation to add
    /// - throws: Panic
    /// - returns: OperationBuilder object after setting
    public func add(control: Operation) -> OperationBuilder {
      TFLib.AddControlInput(descriptor, control.operation)
      return self
    }//end add

    /// build the operation
    /// - throws: Panic
    /// - returns: a new operation object
    public func build() throws -> Operation {
      let s = try Status()
      let op = TFLib.FinishOperation(descriptor, s.status)
      guard s.code == .OK else { throw Panic.FAULT(reason: s.message) }
      return Operation(op)
    }//end func

    /// set the device
    /// - parameters:
    ///   - device: device name of the operation
    /// - throws: Panic
    /// - returns: OperationBuilder object after setting
    public func `set`(device: String) -> OperationBuilder {
      TFLib.SetDevice(descriptor, device)
      return self
    }//end func

    /// set attributes for the operation to build
    /// - parameters:
    ///   - attributes: a dictionary of attributes of the operation, key for the attribute name. Valid attributes include Int64, [Int64], Float, [Float], Bool, [Bool], DataType, [DataType], String, [String], Shape, [Shape], Tensor, [Tensor], TensorProto, [TensorProto], Data
    /// - throws: Panic
    /// - returns: OperationBuilder object after setting
    public func `set`(attributes: [String: Any] = [:]) throws -> OperationBuilder {
      var total = 0
      for (k, v) in attributes {

        if v is Int64, let i = v as? Int64 {
          TFLib.SetAttrInt(descriptor, k, i)
          total += 1
        } else if v is [Int64], let iv = v as? [Int64], iv.count > 0 {
          iv.withUnsafeBufferPointer { ptr in
            if let array = ptr.baseAddress {
              TFLib.SetAttrIntList(descriptor, k, array, Int32(iv.count))
              total += 1
            }//end if
          }//end if
        } else if v is Double, let d = v as? Double {
          let f = Float(d)
          TFLib.SetAttrFloat(descriptor, k, f)
          total += 1
        } else if v is [Double], let dv = v as? [Double], dv.count > 0 {
          let fv = dv.map { Float( $0) }
          fv.withUnsafeBufferPointer { ptr in
            if let array = ptr.baseAddress {
              TFLib.SetAttrFloatList(descriptor, k, array, Int32(fv.count))
              total += 1
            }//end if
          }//end pointer
        } else if v is Float, let f = v as? Float {
          TFLib.SetAttrFloat(descriptor, k, f)
          total += 1
        } else if v is [Float], let fv = v as? [Float], fv.count > 0 {
          fv.withUnsafeBufferPointer { ptr in
            if let array = ptr.baseAddress {
              TFLib.SetAttrFloatList(descriptor, k, array, Int32(fv.count))
              total += 1
            }//end if
          }//end pointer
        } else if v is Bool, let b = v as? Bool {
          let bit: UInt8 = b ? 1 : 0
          TFLib.SetAttrBool(descriptor, k, bit)
          total += 1
        } else if v is [Bool], let bv = v as? [Bool], bv.count > 0 {
          let bits = bv.map { $0 ? UInt8(1) : UInt8(0) }
          bits.withUnsafeBufferPointer { ptr in
            if let array = ptr.baseAddress {
              TFLib.SetAttrBoolList(descriptor, k, array, Int32(bits.count))
              total += 1
            }//end if
          }//end pointer
        } else if v is DataType, let d = v as? DataType {
          TFLib.SetAttrType(descriptor, k, Int32(d.rawValue))
          total += 1
        } else if v is [DataType], let dv = v as? [DataType], dv.count > 0 {
          let dt = dv.map { Int32($0.rawValue) }
          dt.withUnsafeBufferPointer { ptr in
            if let array = ptr.baseAddress {
              TFLib.SetAttrTypeList(descriptor, k, array, Int32(dt.count))
              total += 1
            }//end if
          }//end pointer
        } else if v is String, let s = v as? String {
          TFLib.SetAttrString(descriptor, k, s, s.utf8.count)
          total += 1
        } else if v is [String], let s = v as? [String], s.count > 0 {
          let pArray:[UnsafePointer<CChar>] = s.map { $0.withCString { $0 } }
          let pLens:[Int] = s.map { $0.utf8.count }
          let array = pArray.withUnsafeBufferPointer { $0.baseAddress }
          let lens = pLens.withUnsafeBufferPointer { $0.baseAddress }
          TFLib.SetAttrStringList(descriptor, k, array, lens, Int32(s.count))
          total += 1
        } else if v is Shape, let s = v as? Shape, s.dimensions.count > 0 {
          s.dimensions.withUnsafeBufferPointer { ptr in
            if let array = ptr.baseAddress {
              TFLib.SetAttrShape(descriptor, k, array, Int32(s.dimensions.count))
              total += 1
            }//end if
          }//end pointer
        } else if v is [Shape], let s = v as? [Shape], s.count > 0 {
          let array = UnsafeMutablePointer<UnsafePointer<Int64>>.allocate(capacity: s.count)
          let lens = UnsafeMutablePointer<Int32>.allocate(capacity: s.count)
          for i in 0 ... s.count - 1 {
            s[i].dimensions.withUnsafeBufferPointer { ptr in
              if let p = ptr.baseAddress {
                array.advanced(by: i).pointee = p
              }//end if
            }//end pointer
            lens.advanced(by: i).pointee = Int32(s[i].dimensions.count)
          }//next
          TFLib.SetAttrShapeList(descriptor, k, array, lens, Int32(s.count))
          total += 1
          lens.deallocate(capacity: s.count)
          array.deallocate(capacity: s.count)
        } else if v is TensorProto, let p = v as? TensorProto {
          let data = try p.serializedData()
          let status = try Status()
          data.withUnsafeBytes { (ptr: UnsafePointer<CChar>) in
            TFLib.SetAttrTensorShapeProto(descriptor, k, ptr, data.count, status.status)
            total += 1
          }//end bytes
          guard status.code == .OK else { throw Panic.FAULT(reason: status.message) }
        } else if v is [TensorProto], let pv = v as? [TensorProto], pv.count > 0 {
          let array = UnsafeMutablePointer<UnsafePointer<CChar>>.allocate(capacity: pv.count)
          let lens = UnsafeMutablePointer<Int>.allocate(capacity: pv.count)
          let data = try pv.map { try $0.serializedData() }
          for i in 0 ... pv.count - 1 {
            data[i].withUnsafeBytes { (ptr: UnsafePointer<CChar>) in
              array.advanced(by: i).pointee = ptr
            }//end bytes
            lens.advanced(by: i).pointee = data[i].count
          }//next
          let status = try Status()
          TFLib.SetAttrTensorShapeProtoList(descriptor, k, array, lens, Int32(pv.count), status.status)
          total += 1
          lens.deallocate(capacity: pv.count)
          array.deallocate(capacity: pv.count)
          guard status.code == .OK else { throw Panic.FAULT(reason: status.message) }
        } else if v is Tensor, let t = v as? Tensor {
          let status = try Status()
          TFLib.SetAttrTensor(descriptor, k, t.tensor, status.status)
          total += 1
          guard status.code == .OK else { throw Panic.FAULT(reason: status.message) }
        }else if v is [Tensor], let tv = v as? [Tensor], tv.count > 0 {
          let pointers = tv.map { $0.tensor }
          let status = try Status()
          pointers.withUnsafeBufferPointer { pointer in
            if let p = pointer.baseAddress {
              TFLib.SetAttrTensorList(descriptor, k, p, Int32(pointers.count), status.status)
              total += 1
            }//end if
          }//end pointers
          guard status.code == .OK else { throw Panic.FAULT(reason: status.message) }
        }else if v is Data, let d = v as? Data, d.count > 0 {
          let status = try Status()
          let p = d.withUnsafeBytes { pointer -> UnsafePointer<Int8> in return pointer }
          TFLib.SetAttrValueProto(descriptor, k, p, d.count, status.status)
          total += 1
          guard status.code == .OK else { throw Panic.FAULT(reason: status.message) }
        }//end if
      }//next
      guard total == attributes.count else
      { throw Panic.FAULT(reason: "Not all attributes have been accepted")}
      return self
    }//end func
  }//end class

  /// Express wrapper of Operation
  public class Operation: Equatable, CustomStringConvertible {
    public let operation: OpaquePointer

    /// compare two operations
    public static func == (opa: Operation, opb: Operation) -> Bool {
      return opa.operation == opb.operation
    }//end func

    /// transform an input to an output
    public static func AsInput(input: Input) -> Output {
      return TFLib.OperationInput(input)
    }//end asInput

    /// TF_AttrType describes the type of the value of an attribute on an operation.
    enum AttrType: Int32 {
      case STRING = 0,
      INT = 1,
      FLOAT = 2,
      BOOL = 3,
      TYPE = 4,
      SHAPE = 5,
      TENSOR = 6,
      PLACEHOLDER = 7,
      FUNC = 8
    }//end enum

    /// lookup an attribute in the current operation
    /// - parameters:
    ///   - forKey: key of the attribute to look for
    /// - returns: attribute value in an AttrValue object
    /// - throws: Panic
    public func attribute(forKey: String) throws -> AttrValue {
      let status = try Status()
      let buffer = try Buffer()
      TFLib.OperationGetAttrValueProto(operation, forKey, buffer.buffer, status.status)
      guard status.code == .OK, let data = buffer.data
        else { throw Panic.FAULT(reason: status.message) }
      return try AttrValue(serializedData: data)
    }//end value

    /// get node definition
    public var nodeDefinition: NodeDef? {
      get {
        do {
          let status = try Status()
          let buffer = try Buffer()
          TFLib.OperationToNodeDef(operation, buffer.buffer, status.status)
          guard status.code == .OK,
            let data = buffer.data
            else { return nil }
          return try NodeDef(serializedData: data)
        }catch {
          return nil
        }//end try
      }//end get
    }//end var

    /// get operation name
    public var name: String? {
      get {
        if let nm = TFLib.OperationName(operation) {
          return String(cString: nm)
        } else {
          return nil
        }//end if
      }//end get
    }//end var

    /// get operation type
    public var `type`: String? {
      get {
        if let tp = TFLib.OperationOpType(operation) {
          return String(cString: tp)
        } else {
          return nil
        }//end if
      }//end get
    }//end var

    /// get device type
    public var device: String {
      get {
        if let dev = TFLib.OperationDevice(operation) {
          return String(cString: dev)
        } else {
          return ""
        }//end if
      }//end get
    }//end var

    public var description: String {
      get {
        let tp = type ?? ""
        let nm = name ?? ""
        let dev = device
        let ni =  numberOfInputs
        let no = numberOfOutputs
        return "Operation: {type = \(tp), name = \(nm), device = \(dev), input = \(ni), output = \(no)}"
      }
    }

    /// get number of inputs of the operation
    public var numberOfInputs: Int {
      get {
        return Int(TFLib.OperationNumInputs(operation))
      }//end get
    }//end var

    /// get number of outputs of the operation
    public var numberOfOutputs: Int {
      get {
        return Int(TFLib.OperationNumOutputs(operation))
      }//end get
    }//end var

    /// get type of the input
    /// - parameters:
    ///   - input: an operation input
    /// returns: data type of the input
    public static func TypeOf(input: Input) -> DataType? {
      let tp = TFLib.OperationInputType(input)
      return DataType(rawValue: Int(tp))
    }//end func

    /// get type of the output
    /// - parameters:
    ///   - output: an operation output
    /// returns: data type of the output
    public static func TypeOf(output: Output) -> DataType? {
      let tp = TFLib.OperationOutputType(output)
      return DataType(rawValue: Int(tp))
    }//end func

    /// get size of the input list array
    /// - parameters:
    ///   - arguments: an argument string
    /// - throws: Panic
    /// - returns: size of the input list
    public func sizeOfInputList(argument: String) throws -> Int {
      let status = try Status()
      let size = Int(TFLib.OperationInputListLength(operation, argument, status.status))
      guard status.code == .OK else { throw Panic.FAULT(reason: status.message) }
      return size
    }//end func

    /// get size of the output list array
    /// - parameters:
    ///   - arguments: an argument string
    /// - throws: Panic
    /// - returns: size of the output list
    public func sizeOfOutputList(argument: String) throws -> Int {
      let status = try Status()
      let size = Int(TFLib.OperationOutputListLength(operation, argument, status.status))
      guard status.code == .OK else { throw Panic.FAULT(reason: status.message) }
      return size
    }//end func

    /// get the consumers of an output
    /// - parameters:
    ///   - output: output to evaluate
    /// - returns: an array of TF_Input
    public static func Consumers(output: Output) -> [Input] {
      let size = Int(TFLib.OperationOutputNumConsumers(output))
      guard size > 0 else { return [Input]() }
      let inputs = UnsafeMutablePointer<Input>.allocate(capacity: size)
      defer {
        inputs.deallocate(capacity: size)
      }//end defer
      let count = TFLib.OperationOutputConsumers(output, inputs, Int32(size))
      guard count == Int32(size) else {
        return [Input]()
      }//end guard
      let buffered = UnsafeMutableBufferPointer<Input>(start: inputs, count: size)
      let array = Array(buffered)
      return array
    }//end Consumers

    /// get control inputs as an operation array
    public var controlInputs: [Operation] {
      let size = Int(TFLib.OperationNumControlInputs(operation))
      guard size > 0 else { return [Operation]() }
      let inputs = UnsafeMutablePointer<OpaquePointer>.allocate(capacity: size)
      defer {
        inputs.deallocate(capacity: size)
      }//end defer
      let count = TFLib.OperationGetControlInputs(operation, inputs, Int32(size))
      guard count == Int32(size) else {
        return [Operation]()
      }//end guard
      let buffered = UnsafeMutableBufferPointer<OpaquePointer>(start: inputs, count: size)
      let array = Array(buffered).map { Operation($0) }
      return array
    }//end controlInputs

    /// get control outputs as an operation array
    public var controlOutputs: [Operation] {
      let size = Int(TFLib.OperationNumOutputs(operation))
      guard size > 0 else { return [Operation]() }

      let outputs = UnsafeMutablePointer<OpaquePointer>.allocate(capacity: size)
      defer {
        outputs.deallocate(capacity: size)
      }//end defer

      let count = TFLib.OperationGetControlOutputs(operation, outputs, Int32(size))
      guard count == Int32(size) else {
        return [Operation]()
      }//end guard
      let buffered = UnsafeMutableBufferPointer<OpaquePointer>(start: outputs, count: size)
      let array = Array(buffered).map { Operation($0) }
      return array
    }//end controlOutputs

    /// - parameters:
    ///   - handle: operation handle
    public init(_ handle: OpaquePointer) {
      operation = handle
    }//end init

    /// - parameters:
    ///   - output: graph output
    public init(_ output: Output) {
      operation = output.oper
    }//end init

    /// - parameters:
    ///   - output: graph output
    public init(_ input: Input) {
      operation = input.oper
    }//end init

    /// generate a TF_Input from the current operation
    public func asInput(_ index: Int) -> Input {
      return TF_Input(oper: operation, index: Int32(index))
    }//end asInput

    /// generate a Output from the current operation
    public func asOutput(_ index: Int) -> Output {
      return output(index)
    }//end asOutput

    /// generate a Output from the current operation
    public func output(_ index: Int) -> Output {
      return TF_Output(oper: operation, index: Int32(index))
    }//end output
  }//end class

  /// Express wrapper of Graph
  public class Graph {
    let graph: OpaquePointer
    var autoDestroy = true

    /// construct a graph from a handle. **NOTE** it will not be deallocated automatically
    /// - parameters:
    ///   - handle: graph handle to copy with
    public init(handle: OpaquePointer) {
      graph = handle
      autoDestroy = false
    }

    /// create a new graph
    public init () throws {
      guard let _ = TFLib.libDLL, let g = TFLib.NewGraph() else {
        throw Panic.CALL
      }//end guard
      graph = g
    }//end init

    deinit {
      if autoDestroy {
        TFLib.DeleteGraph(graph)
      }//end if
    }//end deinit

    /// Generate an OperationBuilder()
    /// - parameters:
    ///   - name: name of the operation
    ///   - type: type of the operation
    /// - throws: Panic
    public func opBuilder(name: String, `type`: String) throws -> OperationBuilder {
      return try OperationBuilder(graph: self, name: name, type: type)
    }//end func

    /// Adds operations to compute the partial derivatives of sum of `y`s w.r.t `x`s,
    /// i.e., d(y_1 + y_2 + ...)/dx_1, d(y_1 + y_2 + ...)/dx_2...
    /// `dx` are used as initial gradients (which represent the symbolic partial
    /// derivatives of some loss function `L` w.r.t. `y`).
    /// `dx` must be nullptr or have size `ny`.
    /// If `dx` is nullptr, the implementation will use dx of `OnesLike` for all
    /// shapes in `y`.
    /// The partial derivatives are returned in `dy`. `dy` should be allocated to
    /// size `nx`.
    ///
    // WARNING: This function does not yet support all the gradients that python
    /// supports. See
    /// https://www.tensorflow.org/code/tensorflow/cc/gradients/README.md
    /// for instructions on how to add C++ more gradients.
    public func addGradients(y: [Output], x: [Output], dx: [Output] = []) throws -> [Output] {
      guard let _ = TFLib.libDLL else { throw Panic.DLL(reason: "Library Not Ready") }
      let s = try Status()
      let countX = x.count
      let countY = y.count
      guard countY > 0 else {
        return [Output]()
      }//end guard
      let dy = UnsafeMutablePointer<Output>.allocate(capacity: countX)
      defer {
        dy.deallocate(capacity: countX)
      }//end dy
      let pY = y.withUnsafeBufferPointer { $0.baseAddress }
      let pX = x.withUnsafeBufferPointer { $0.baseAddress }

      var pdx = UnsafePointer<Output>(bitPattern: 0)
      if dx.count > 0 {
        guard dx.count == countY else { throw Panic.FAULT(reason: "dx must have size of y")}
        pdx = dx.withUnsafeBufferPointer { $0.baseAddress }
      }//end if

      TFLib.AddGradients(graph, pY, Int32(countY), pX, Int32(countX), pdx, s.status, dy)

      guard s.code == .OK else { throw Panic.FAULT(reason: s.message) }
      let pdy = UnsafeBufferPointer(start: dy, count: countY)
      return Array(pdy)
    }

    /// set a tensor with a specific shape
    /// - parameters:
    ///   - output: an output of current graph
    ///   - shape: dimensions of the output
    /// - throws: Panic
    public func setTensor(output: Output, shape: [Int64]) throws {
      let status = try Status()
      guard shape.count > 0,
        let p = (shape.withUnsafeBufferPointer {
          ptr -> UnsafePointer<Int64>? in return ptr.baseAddress })
        else { throw Panic.INVALID }
      TFLib.GraphSetTensorShape(graph, output, p, Int32(shape.count), status.status)
      guard status.code == .OK else { throw Panic.FAULT(reason: status.message) }
    }//end setTensor

    /// get the tensor shape dimensions.
    /// - parameters:
    ///   - output: an output of current graph
    /// - throws: Panic
    /// - returns: dimensions of the output
    public func getTensor(output: Output) throws -> [Int64] {
      let status = try Status()
      let pStatus = status.status
      let size = Int(TFLib.GraphGetTensorNumDims(graph, output, pStatus))
      guard size > 0 else { throw Panic.FAULT(reason: "Shape has no dimensions") }

      let array = UnsafeMutablePointer<Int64>.allocate(capacity: size)
      defer {
        array.deallocate(capacity: size)
      }//end defer

      TFLib.GraphGetTensorShape(graph, output, array, Int32(size), pStatus)
      guard status.code == .OK else {
        throw Panic.FAULT(reason: status.message)
      }//end guard
      let dimbuf = UnsafeMutableBufferPointer<Int64>(start: array, count: size)
      let shape = Array(dimbuf)
      return shape
    }//end getTensor

    /// search for a specific operation by name
    public func searchOperation(forName: String) throws -> Operation {
      guard let op = TFLib.GraphOperationByName(graph, forName) else {
        throw Panic.INVALID
      }//end guard
      return Operation(op)
    }//end func

    /// retrieve all operations into an array
    public var operations: [Operation] {
      get {
        var cursor = 0
        var ops = [Operation]()
        while let pointer = TF_NextGraphOperation(graph, &cursor) {
          ops.append(Operation(pointer))
        }//next
        return ops
      }//end get
    }//end var

    /// get the definition proto buffer
    public var definition: GraphDef? {
      get {
        guard let data = self.buffer?.data else {
          return nil
        }//end guard

        do {
          let def = try GraphDef(serializedData: data)
          return def
        } catch {
          return nil
        }//end try
      }//end get
    }//end var

    /// get the definition as a buffer
    public var buffer: Buffer? {
      get {
        do {
          let buffer = try Buffer()
          let status = try Status()
          TFLib.GraphToGraphDef(graph, buffer.buffer, status.status)
          guard status.code == .OK else { return nil }
          return buffer
        } catch {
          return nil
        }//end try
      }//end get
    }//end buffer

    /// import a proto buffer with options
    /// - parameters:
    ///   - buf: proto buffer for the graph definition
    ///   - options: optoins to apply with
    /// - throws: Panic
    public func `import`(buf: Buffer, options: GraphDefOptions? = nil) throws {
      let status = try Status()
      let op: GraphDefOptions
      if let o = options {
        op = o
      } else {
        op = try GraphDefOptions()
      }//end if
      TFLib.GraphImportGraphDef(graph, buf.buffer, op.options, status.status)
      guard status.code == .OK else { throw Panic.FAULT(reason: status.message) }
    }//end func

    /// import a definition with options
    /// - parameters:
    ///   - definition: GraphDef data object
    ///   - options: optoins to apply with
    /// - throws: Panic
    public func `import`(definition: GraphDef, options: GraphDefOptions? = nil) throws {
      let buf = try Buffer(data: try definition.serializedData())
      try self.import(buf: buf, options: options)
    }//end import

    /// import a definition buffer and return outputs
    /// import a graph definition with returning outputs
    /// - parameters:
    ///   - buf: proto buffer for the graph definition
    ///   - options: optoins to apply with
    /// - throws: Panic
    /// - returns: an array of outputs
    public func importWithReturnOutputs(buf: Buffer, options: GraphDefOptions) throws -> [ Output ] {
      let status = try Status()
      let count = Int(TFLib.ImportGraphDefOptionsNumReturnOutputs(options.options))
      if count < 1 {
        let pOutputs = UnsafeMutablePointer<Output>.allocate(capacity: 1)
        TFLib.GraphImportGraphDefWithReturnOutputs(graph, buf.buffer, options.options, pOutputs, 0, status.status)
        pOutputs.deallocate(capacity: 1)
        guard let code = status.code, code == .OK else {
          throw Panic.FAULT(reason: status.message)
        }//end guard
        return [ Output ]()
      }//end if

      let pOutputs = UnsafeMutablePointer<Output>.allocate(capacity: count)
      defer { pOutputs.deallocate(capacity: count) }
      TFLib.GraphImportGraphDefWithReturnOutputs(graph, buf.buffer, options.options, pOutputs, Int32(count),  status.status)
      guard status.code == .OK else {
        throw Panic.FAULT(reason: status.message)
      }//end guard
      let buffered = UnsafeMutableBufferPointer<Output>(start: pOutputs, count: count)
      let outputs = Array(buffered)
      return outputs
    }//end func

    /// import a graph definition with returning outputs
    /// - parameters:
    ///   - definition: GraphDef data object
    ///   - options: optoins to apply with
    /// - throws: Panic
    /// - returns: an array of outputs
    public func importWithReturnOutputs(definition: GraphDef, options: GraphDefOptions) throws -> [Output] {
      let buf = try Buffer(data: try definition.serializedData())
      return try self.importWithReturnOutputs(buf: buf, options: options)
    }//end func

    /// create a new session from current graph
    /// - parameters:
    ///   - options: session options, optional.
    /// - returns: a new session object
    /// - throws: Panic
    public func newSession(options: SessionOptions? = nil) throws -> Session {
      return try Session(graph: self, options: options)
    }//end session

    public func runner() throws -> Runner {
      return try Runner(graph: self)
    }//end runner

    /// This function creates a new TF_Session (which is created on success) using session_options, and then initializes state (restoring tensors and other assets) using run_options.
    /// Any NULL and non-NULL value combinations for (run_options,meta_graph_def`) are valid.
    /// - parameters:
    ///   - sessionOptions: session Options
    ///   - exportDir: must be set to the path of the exported SavedModel.
    ///   - tags: must include the set of tags used to identify one MetaGraphDef in the SavedModel.
    /// - returns: a new session object
    /// - throws: Panic
    public func load(sessionOptions: SessionOptions? = nil, runOptions: Buffer? = nil, exportDir: String, tags: [String], metaGraphDef: Buffer) throws -> Runner {
      return try Runner(graph: self, sessionOptions: sessionOptions, runOptions: runOptions, exportDir: exportDir, tags: tags, metaGraphDef: metaGraphDef)
    }//end session

    /// Generate a const operation
    /// - parameters:
    ///   - tensor: Tensor to build in this operation
    ///   - type: String, default is "Const"
    ///   - name: String, default is "const". **CAUTION:** Naming space is not available up to 1.3.0
    /// - returns: Operation
    /// - throws: Panic
    public func const(tensor: Tensor, `type`: String = "Const", name: String = "const") throws -> Operation {
      guard let tp = tensor.type else { throw Panic.INVALID }
      return try self.opBuilder(name: name, type: type)
        .set(attributes: ["value": tensor, "dtype": tp]).build()
    }//end const

    /// Generate a placeholder tensor
    /// - parameters:
    ///   - type: String, default is "Placeholder"
    ///   - name: String, default is "feed". **CAUTION:** Naming space is not available up to 1.3.0
    /// - returns: Operation
    /// - throws: Panic
    public func placeholder(`type`: String = "Placeholder", name: String = "feed") throws -> Operation {
      return try self.opBuilder(name: name, type: type)
        .set(attributes: ["dtype": DataType.dtInt32]).build()
    }//end Placeholder

    /// Generate an integer scalar tensor
    /// - parameters:
    ///   - v: Integer
    ///   - name: String, default is "scalar". **CAUTION:** Naming space is not available up to 1.3.0
    /// - returns: Operation
    /// - throws: Panic
    public func scalar(_ v: Int, name: String = "scalar") throws -> Operation {
      let x = try Tensor.Scalar(Int32(v))
      return try self.const(tensor: x, name: name)
    }//end ScalarConst

    /// Generate an float type scalar tensor
    /// - parameters:
    ///   - v: Float
    ///   - name: String, default is "scalar". **CAUTION:** Naming space is not available up to 1.3.0
    /// - returns: Operation
    /// - throws: Panic
    public func scalar(_ v: Float, name: String = "scalar") throws -> Operation {
      let x = try Tensor.Scalar(Float32(v))
      return try self.const(tensor: x, name: name)
    }//end ScalarConst

    /// Add two inputs
    /// - parameters:
    ///   - left: Output, left input to add
    ///   - right: Output, right input to add
    ///   - type: String, type of the operation, default is "AddN"
    ///   - name: String, default is "add". **CAUTION:** Naming space is not available up to 1.3.0
    /// - returns: Operation
    /// - throws: Panic
    public func add(left: Output, right: Output, `type`: String = "AddN", name: String = "add") throws -> Operation {
      return try self.opBuilder(name: name, type: type).add(inputs: [left, right]).build()
    }//end Add

    /// Add two operations
    /// - parameters:
    ///   - left: Operation, left operation to add
    ///   - right: Operation, right operation to add
    ///   - name: String, default is "add". **CAUTION:** Naming space is not available up to 1.3.0
    /// - returns: Operation
    /// - throws: Panic
    public func add(left: Operation, right: Operation, name: String = "add") throws -> Operation {
      return try self.add(left: left.asOutput(0), right: right.asOutput(0), name: name)
    }//end Add

    /// Generate a negative operation from the current one.
    /// - parameters:
    ///   - n: Operation, the original operation.
    ///   - type: String, type of the operation, default is "Neg"
    ///   - name: String, default is "neg". **CAUTION:** Naming space is not available up to 1.3.0
    /// - returns: Operation
    /// - throws: Panic
    public func neg(_ n: Operation, `type`: String = "Neg", name: String = "neg") throws -> Operation {
      return try self.opBuilder(name: name, type: type)
        .add(input: n.asOutput(0)).build()
    }//end neg

    /// Compare two inputs and test if the left operand is less than the right one.
    /// - parameters:
    ///   - left: Output, left input to compare
    ///   - right: Output, right input to compare
    ///   - type: String, type of the operation, default is "Less"
    ///   - name: String, default is "less_than". **CAUTION:** Naming space is not available up to 1.3.0
    /// - returns: Operation
    /// - throws: Panic
    public func lessThan(left: Output, right: Output, `type`: String = "Less", name: String = "less_than") throws -> Operation {
      return try self.opBuilder(name: name, type: type).add(input: left).add(input: right).build()
    }//end LessThan

    /// Setup a 2 x 2 matrix
    /// - parameters:
    ///   - values: [Float], the matrix in row major order.
    ///   - name: String, name of the matrix to create
    /// - returns: Operation
    /// - throws: Panic
    public func floatConst2x2(values: [Float], name: String) throws -> Operation {
      let tensor = try Tensor.Array(dimensions: [2,2], value: values)
      return try self.opBuilder(name: name, type: "Const")
        .set(attributes: ["value": tensor, "dtype": DataType.dtFloat]).build()
    }//end FloatConst2x2

    /// Matrix Multipliction
    /// - parameters:
    ///   - l: Operation, left matrix to multiply
    ///   - r: Operation, right matrix to multiply
    ///   - name: String, name of the matrix to create
    ///   - transposeA: Bool, should transpose left matrix before multiplication
    ///   - transposeB: Bool, should transpose right matrix before multiplication
    /// - returns: Operation
    /// - throws: Panic
    public func matMul(l: Operation, r: Operation, name: String, transposeA: Bool = false, transposeB: Bool = false ) throws -> Operation {
      var a: [String: Any] = [:]
      if transposeA {
        a["transpose_a"] = true
      }//end if
      if transposeB {
        a["transpose_b"] = true
      }//end if
      return try self.opBuilder(name: name, type: "MatMul").set(attributes: a)
        .add(input: l.output(0))
        .add(input: r.output( 0))
        .build()
    }//end MatMul

    public func OnesLike(inp: Operation, name: String) throws -> Operation {
      return try self.opBuilder(name: name, type: "OnesLike")
        .add(input: inp.output(0)).build()
    }//end OnesLike

    public func GradientOp(inp: Operation, name: String) throws -> Operation {
      return try self.opBuilder( name: name, type: "TestOpWithNoGradient")
        .add(input: inp.output(0)).build()
    }//end NoGradientOp


    public func div(x: Output, y: Output, name: String = "", index: Int = 0) throws -> Output {
      return try self.binaryOp("Div", x, y, name: name, index: index)
    }

    public func sub(x: Output, y: Output, name: String = "", index: Int = 0) throws -> Output {
      return try self.binaryOp("Sub", x, y, name: name, index: index)
    }

    public func resizeBilinear(images: Output, size: Output, name: String = "", index: Int = 0) throws -> Output {
      return try self.binaryOp("ResizeBilinear", images, size, name: name, index: index)
    }

    public func expandDims(input: Output, dim: Output, name: String = "", index: Int = 0) throws -> Output {
      return try self.binaryOp("ExpandDims", input, dim, name:name, index: index)
    }

    public func cast(value: Output, dtype: DataType, name: String = "Cast", `type`: String = "Cast", index:Int = 0, keyName: String = "DstT") throws -> Output {
      return try self.opBuilder(name: name, type: type)
        .add(input: value).set(attributes: [keyName: dtype]).build().output(index)
    }

    public func decodeJpeg(content: Output, channels: Int, index:Int = 0, name: String = "DecodeJpeg", `type`: String = "DecodeJpeg", keyName: String = "channels") throws -> Output {
      return try self.opBuilder(name: name, type: type)
        .add(input: content)
        .set(attributes: [keyName: Int64(channels)])
        .build().output(index)
    }

    public func constant<T>(name: String, value: T, index:Int = 0) throws -> Output {
      let t = try Tensor.Scalar(value)
      return try self.const(tensor: t, name: name).asOutput(index)
    }

    public func constantArray<T>(name: String, value: [T], index:Int = 0) throws -> Output {
      let t = try Tensor.Array(dimensions: [Int64(value.count)], value: value)
      return try self.const(tensor: t, name: name).asOutput(index)
    }

    public func binaryOp(_ `type`: String, _ in1: Output, _ in2: Output, name: String = "", index: Int = 0) throws -> Output {
        let nm = name.isEmpty ? type: name
        return try self.opBuilder(name: nm, type: type)
          .add(input: in1).add(input: in2).build().output(index)
    }

    /// Create a function from a graph
    /// - parameters:
    ///   - name: String, the name of the new TF_Function. Should match the operation name (OpDef.name) regexp [A-Z][A-Za-z0-9_.\\-/]* and be distinct from other operation names (at least those registered in graphs  where this function will be used).
    ///   - operations: [Operation], Array of operations to become the body of the function or null
    ///   - inputs: [Output], array of TF_Outputs that specify the inputs to the function
    ///   - outputs: [Output], array of TF_Outputs that specify the outputs of the function.
    ///   - outputNames: [String], The names of the function's outputs. Must either have the same length as `outputs` or be null. In the former case, the names should match the regular expression for ArgDef names - "[a-z][a-z0-9_]*". In the latter case, names for outputs will be generated automatically.
    public func toFunction(_ name: String, operations: [Operation], inputs: [Output], outputs: [Output], outputNames: [String], options: OpaquePointer? = nil) throws -> Function? {
      guard outputs.count == outputNames.count else {
        throw Panic.FAULT(reason: "Output array elements are mismatched with names")
      }
      let status = try Status()
      let opera:UnsafePointer<OpaquePointer?>? = operations.map { $0.operation }
        .withUnsafeBufferPointer { $0.baseAddress }
      let pInputs = inputs.withUnsafeBufferPointer { $0.baseAddress }
      let pOutpus = outputs.withUnsafeBufferPointer { $0.baseAddress }
      let pOutputNames:UnsafePointer<UnsafePointer<CChar>?>? = outputNames
        .map { $0.withCString { p -> UnsafePointer<CChar> in return p } }
        .withUnsafeBufferPointer { $0.baseAddress }
      guard let fun = TFLib.GraphToFunction(graph, name, Int32(operations.count > 0 ? operations.count: -1), operations.count > 0 ? opera : nil, Int32(inputs.count), pInputs, Int32(outputs.count), pOutpus, pOutputNames, options, status.status),
      let code = status.code, code == .OK else {
        throw Panic.FAULT(reason: status.message)
      }//end guard
      return Function(self, reference: fun)
    }

    /// Function is a grouping of operations with defined inputs and outputs.
    /// Once created and added to graphs, functions can be invoked by creating an
    /// operation whose operation type matches the function name.
    public class Function {
      let g: Graph
      let ref: OpaquePointer

      /// constructor. DO **NOT** CALL IT DIRECTLY. Call `Graph.toFunction()` to generate function instead.
      public init(_ graph: Graph, reference: OpaquePointer) {
        g = graph
        ref = reference
      }

      /// Add `function` to graph `g`. Once `function` is added to `g`,
      /// it can be called by creating an operation using the function's name.
      public func add () throws {
        let status = try Status()
        TFLib.GraphAddFunction(g.graph, ref, status.status)
        guard let code = status.code, code == .OK else {
          throw Panic.FAULT(reason: status.message)
        }//end guard
      }

      /// delete function
      public func delete() {
        TFLib.DeleteFunction(ref)
      }

      /// get protocol buffer of the current function
      public var buffer: Buffer? {
        guard let buf = try? Buffer(), let status = try? Status() else {
          return nil
        }
        TFLib.FunctionToFunctionDef(ref, buf.buffer, status.status)
        if let code = status.code, code == .OK {
          return buf
        } else {
          return nil
        }
      }
    }

  }//end graph

  /// class wrapper of Graph Definition Options
  public class GraphDefOptions {

    let options: OpaquePointer

    /// Generate a new options object
    public init() throws {
      guard let _ = TFLib.libDLL,
        let op = TFLib.NewImportGraphDefOptions()
      else { throw Panic.CALL }
      options = op
    }//end init

    deinit {
      TFLib.DeleteImportGraphDefOptions(options)
    }//end init

    /// Set the prefix to be prepended to the names of nodes in graph_def that will be imported into graph.
    /// - parameters:
    ///   - prefix: the prefix to add to all the names of nodes.
    public func `set`(prefix: String) {
      TFLib.ImportGraphDefOptionsSetPrefix(options, prefix)
    }//end func

    /// Set any imported nodes with input src_name:src_index to have that input replaced with dst. src_name refers to a node in the graph to be imported, dst references a node already existing in the graph being imported into.
    /// - parameters:
    ///   - sourceName: source operation name
    ///   - sourceIndex: source operation index
    ///   - destination: destinated output to map
    public func addInputMapping(sourceName: String, sourceIndex: Int, destination: Output) {
      TFLib.ImportGraphDefOptionsAddInputMapping(options, sourceName, Int32(sourceIndex), destination)
    }//end func

    /// Set any imported nodes with control input src_name to have that input replaced with dst. src_name refers to a node in the graph to be imported, dst references an operation already existing in the graph being imported into.
    /// - parameters:
    ///   - source: source operation
    ///   - destination: the destinated control to remap
    public func remapControlDependency(source: String, destination: Operation) {
      TFLib.ImportGraphDefOptionsRemapControlDependency(options, source, destination.operation)
    }//end func

    /// Add an output in graph_def to be returned via the return_outputs output parameter of TF_GraphImportGraphDef(). If the output is remapped via an input mapping, the corresponding existing tensor in graph will be returned.
    /// - parameters:
    ///   - operationName: name of the operation to add return output
    ///   - index: index of the return output
    public func addReturnOutput(operationName: String, index: Int) {
      TFLib.ImportGraphDefOptionsAddReturnOutput(options, operationName, Int32(index))
    }//end func

    /// Cause the imported graph to have a control dependency on oper. oper should exist in the graph being imported into.
    /// - parameters:
    ///   - operation: operation to add the control dependency
    public func addControlDependency(operation: Operation) {
      TFLib.ImportGraphDefOptionsAddControlDependency(options, operation.operation)
    }//end func
  }//end class

  /// Class wrapper of TF_WhileParams
  public class GraphWhile {

    var param: TF_WhileParams
    var unloaded = false

    /// Creates a TF_WhileParams for creating a while loop in g. inputs are outputs that already exist in g used as initial values for the loop variables.The returned TF_WhileParams will have all fields initialized except cond_output, body_outputs, and name. The body_outputs buffer will be allocated to size ninputs. The caller should build cond_graph and body_graph starting from the inputs, and store the final outputs in cond_output and body_outputs.
    /// - parameters:
    ///   - graph: parent graph
    ///   - inputs: inputs for the graph
    public init(graph: Graph, inputs: [Output]) throws {
      guard let _ = TFLib.libDLL else { throw Panic.DLL(reason: "Library is Missing") }
      let status = try Status()
      param = try inputs.withUnsafeBufferPointer { ptr -> TF_WhileParams in
        guard let pointer = ptr.baseAddress else { throw Panic.INVALID }
        return TFLib.NewWhile(graph.graph, pointer, Int32(inputs.count), status.status)
      }//end try
      guard status.code == .OK else { throw Panic.FAULT(reason: status.message) }
    }//end init

    /// Builds the while loop specified by params and returns the output tensors of the while loop in outputs. outputs should be allocated to size params.ninputs. params is no longer valid once this returns. Either this or TF_AbortWhile() must be called after a successful TF_NewWhile() call.
    public func finish() throws -> [Output] {
      let status = try Status()
      let size = Int(param.ninputs)
      guard size > 0 else
      { throw Panic.FAULT(reason: "Unexpect params.ninputs = \(size)") }

      let outputs = UnsafeMutablePointer<Output>.allocate(capacity: size)
      defer {
        outputs.deallocate(capacity: size)
      }//end defer
      TFLib.FinishWhile(&param, status.status, outputs)

      guard status.code == .OK else{
        throw Panic.FAULT(reason: status.message)
      }//end guard

      let array = Array(UnsafeMutableBufferPointer<Output>(start: outputs, count: size))

      unloaded = true
      return array
    }//end func

    /// Frees paramss resources without building a while loop. params is no longer valid after this returns. Either this or TF_FinishWhile() must be called after a successful TF_NewWhile() call.
    public func abort() {
      TFLib.AbortWhile(&param)
      unloaded = true
    }//end abort

    deinit {
      if unloaded { return }
      abort()
    }
  }//end class

  /// get all registered operations
  public class OperationList {

    /// storage of all operations
    public let operations: [OpDef]
    public init() throws {
      guard let _ = TFLib.libDLL, let buf = TFLib.GetAllOpList() else { throw Panic.CALL }
      let pb = Buffer(buf: buf)
      guard let dat = pb.data else { throw Panic.FAULT(reason: "Buffer contains no valid data") }
      let opList = try Tensorflow_OpList(serializedData: dat)
      operations = opList.op
    }
  }

  /// Tensor Session Object
  public class Session {

    let session: OpaquePointer

    /// return all devices for this session in a dictionary.
    /// each key in the dictionary represents a device name,
    /// and the value is a tuple of device type and its memory size, in bytes
    public var devices: [String:(`type`:String, memory: Int64)] {
      var dev: [String:(`type`:String, memory: Int64)] = [:]

      guard let _ = TFLib.libDLL, let status = try? Status(),
        let list = TFLib.SessionListDevices(session, status.status),
        status.code == .OK else { return dev }

      defer { TFLib.DeleteDeviceList(list) }
      for i in 0 ..< TFLib.DeviceListCount(list) {
        if let nm = TFLib.DeviceListName(list, i, status.status),
          status.code == .OK,
          let tp = TFLib.DeviceListType(list, i, status.status),
          status.code == .OK {
          let mem = TFLib.DeviceListMemoryBytes(list, i, status.status)
          if status.code == .OK,
            let dname = String(utf8String: nm),
            let dtype = String(utf8String: tp) {
            dev[dname] = (type: dtype, memory: mem)
          }
        }
      }
      return dev
    }

    /// Return a new execution session with the associated graph, or NULL on error. *graph must be a valid graph (not deleted or nullptr). This function will prevent the graph from being deleted until TF_DeleteSession() is called. Does not take ownership of opts.
    /// - parameters:
    ///   - graph: the parent graph
    ///   - options: options to apply to this session. Optional
    public init(graph: Graph, options: SessionOptions? = nil) throws {
      let status = try Status()
      guard let _ = TFLib.libDLL else { throw Panic.DLL(reason: "Library is Missing") }
      let defOptions = try SessionOptions()
      let opt: OpaquePointer
      if let opts = options {
        opt = opts.options
      } else {
        opt = defOptions.options
      }//end if
      guard let s = TFLib.NewSession(graph.graph, opt, status.status),
        status.code == .OK else
      { throw Panic.FAULT(reason: status.message) }
      session = s
    }//end init

    /// This function creates a new TF_Session (which is created on success) using session_options, and then initializes state (restoring tensors and other assets) using run_options.
    /// Any NULL and non-NULL value combinations for (run_options,meta_graph_def`) are valid.
    /// - parameters:
    ///   - sessionOptions: session options
    ///   - runOptions: run options in a buffer
    ///   - exportDir: must be set to the path of the exported SavedModel.
    ///   - tags: must include the set of tags used to identify one MetaGraphDef in the SavedModel.
    ///   - graph must be a graph newly allocated with TF_NewGraph().
    ///   - metaGraphDef: the meta data to return
    /// - If successful, populates graph with the contents of the Graph and meta_graph_def with the MetaGraphDef of the loaded model.
    public init (sessionOptions: SessionOptions?, runOptions: Buffer?, exportDir: String, tags: [String], graph: Graph, metaGraphDef: Buffer) throws {
      let ops: SessionOptions
      let opr: Buffer
      if let oa = sessionOptions {
        ops = oa
      } else {
        ops = try SessionOptions()
      }//end oa
      if let ob = runOptions {
        opr = ob
      }else {
        opr = try Buffer()
      }//end ob
      let status = try Status()
      guard let _ = TFLib.libDLL else { throw Panic.DLL(reason: "Library is Missing") }
      let tagPointers = tags.map { strdup($0) }
      defer { tagPointers.forEach { free($0) } }
      guard
        let pointer = (tagPointers.withUnsafeBufferPointer { ptr -> UnsafePointer<UnsafeMutablePointer<CChar>?>? in return ptr.baseAddress }),
        let s = TFLib.LoadSessionFromSavedModel (ops.options, opr.buffer, exportDir, pointer, Int32(tags.count), graph.graph, metaGraphDef.buffer, status.status),
        status.code == .OK else {
          throw Panic.FAULT(reason: status.message)
      }//end guard

      session = s
    }//end init

    deinit {
      do {
        let status = try Status()
        TFLib.DeleteSession(session, status.status)
        if status.code != .OK {
          print("delete session fault: \(status.message)")
        }//end if
      }catch {
        print("delete session: \(error)")
      }//end try
    }//end deinit

    /// close the session
    public func close() throws {
      let status = try Status()
      TFLib.CloseSession(session, status.status)
      guard status.code == .OK else { throw Panic.FAULT(reason: status.message) }
    }//end deinit

    /// Run the graph associated with the session starting with the supplied inputs
    /// (inputs[0,ninputs-1] with corresponding values in input_values[0,ninputs-1]).
    ///
    /// Any NULL and non-NULL value combinations for (`run_options`,
    /// `run_metadata`) are valid.
    ///
    ///    - `run_options` may be NULL, in which case it will be ignored; or
    ///      non-NULL, in which case it must point to a `TF_Buffer` containing the
    ///      serialized representation of a `RunOptions` protocol buffer.
    ///    - `run_metadata` may be NULL, in which case it will be ignored; or
    ///      non-NULL, in which case it must point to an empty, freshly allocated
    ///      `TF_Buffer` that may be updated to contain the serialized representation
    ///      of a `RunMetadata` protocol buffer.
    ///
    /// The caller retains ownership of `input_values` (which can be deleted using
    /// TF_DeleteTensor). The caller also retains ownership of `run_options` and/or
    /// `run_metadata` (when not NULL) and should manually call TF_DeleteBuffer on
    /// them.
    ///
    /// On success, the tensors corresponding to outputs[0,noutputs-1] are placed in
    /// output_values[]. Ownership of the elements of output_values[] is transferred
    /// to the caller, which must eventually call TF_DeleteTensor on them.
    ///
    /// On failure, output_values[] contains NULLs.
    /// - parameters:
    ///   - inputs: array of (Output, Tensor) turples. Optional
    ///   - outputs: array of Outputs. Optional
    ///   - targets: array of target operations. Optional
    ///   - options: Buffer of RunOptions Protobuf. Optional
    ///   - metaData: Buffer of RunMetadata Protobuf. Optional
    /// - throws: Panic
    /// - returns: Array of Tensor for the result of session run.
    public func run
      (inputs: [(Output, Tensor)]? = nil,
       outputs: [Output]? =  nil, targets: [Operation]? = nil,
       options: Buffer? = nil, metaData: Buffer? = nil
      ) throws -> [Tensor]
    {
      guard let _ = TFLib.libDLL else { throw Panic.DLL(reason: "Lib Not Ready")}

      let s = try Status()

      let szInput: Int

      if let size = inputs?.count, size > 0 {
        szInput = size
      } else {
        szInput = 0
      }//end if

      let szOutput: Int
      let pOutputValues: UnsafeMutablePointer<OpaquePointer>?
      if let size = outputs?.count, size > 0 {
        szOutput = size
        pOutputValues = UnsafeMutablePointer<OpaquePointer>.allocate(capacity: szOutput)
      } else {
        pOutputValues = UnsafeMutablePointer<OpaquePointer>(bitPattern: 0)
        szOutput = 0
      }

      let szTargets: Int
      if let size = targets?.count, size > 0 {
        szTargets = size
      } else {
        szTargets = 0
      }//end if

      let inpVar = inputs?.map { $0.0 }
      let inpVal = inputs?.map { $0.1.tensor }
      let pTargets = targets?.map { $0.operation }
      TFLib.SessionRun(
        session, options?.buffer,

        inpVar?.withUnsafeBufferPointer { $0.baseAddress },
        inpVal?.withUnsafeBufferPointer { $0.baseAddress },
        Int32(szInput),

        outputs?.withUnsafeBufferPointer { $0.baseAddress },
        pOutputValues,
        Int32(szOutput),

        pTargets?.withUnsafeBufferPointer { $0.baseAddress },
        Int32(szTargets),
        metaData?.buffer, s.status)

      let outputTensors: [Tensor]

      if szOutput > 0,  let p = pOutputValues {
        let values = UnsafeMutableBufferPointer<OpaquePointer>(start: p, count: szOutput)

        // WARNING: tensors created here are WILD - autodestroy = false
        outputTensors = Array(values).map { Tensor(handle: $0) }
        p.deallocate(capacity: szOutput)
      } else {
        outputTensors = [Tensor]()
      }//end if

      guard s.code == .OK else { throw Panic.FAULT(reason: s.message) }

      // WARNING: all tensors created by array must set autodestroy to true!!!
      outputTensors.forEach { $0.autoDestroy = true }
      return outputTensors
    }//end fun

    /// generate a partial session
    public func partial( inputs: [Output]? = nil, outputs: [Output]? = nil, targets: [Operation]? = nil) throws -> PartialSession {
      return try PartialSession(session: self, inputs: inputs, outputs: outputs, targets: targets)
    }//end partial

  }//end class

  /// Run Operations and evaluate Tensors.
  /// A Runner runs the necessary graph fragments to execute every Operation required to
  /// evaluate the Tensors to fetch. The feed(String,int,Tensor) call allows callers
  /// to override the value of Tensors in the graph by substituing the provided
  /// Tensors for the outputs of the operations provided to feed(String,int,Tensor).
  public final class Runner {
    var inputs:[(Output, Tensor)] = []
    var outputs: [Output] = []
    var targets: [Operation] = []
    var runOptions: Buffer?
    let session: Session
    let g: Graph

    /// Use t instead of the Tensor referred to by executing the operation referred to by output.
    public func feed(_ output: Output, tensor: Tensor) -> Runner {
      inputs.append((output, tensor))
      return self
    }//end feed

    /**
     * Avoid evaluating operation and substitute tensor for the value it produces.
     *
     */
    public func feed(_ operation: Operation, index: Int = 0, tensor: Tensor) -> Runner {
      inputs.append((operation.asOutput(index), tensor))
      return self
    }//end feed

    /**
     * Avoid evaluating the index-th output of operation by substituting tensor
     * for the value it produces.
     *
     * Operations in a Graph can have multiple outputs, index identifies which
     * one tensor is being provided for.
     */
    public func feed(_ operation: String, index: Int = 0, tensor: Tensor) throws -> Runner {
      let op = try g.searchOperation(forName: operation)
      return feed(op, index: index, tensor: tensor)
    }//end feed

    /** Makes run() return the Tensor referred to by output. */
    public func fetch(_ output: Output) -> Runner {
      outputs.append(output)
      return self
    }//end fetch

    /// Make run() return the output of operation.
    public func fetch(_ operation: Operation, index: Int = 0) -> Runner {
      return self.fetch(operation.asOutput(index))
    }//end fetch

    /// Make run() return the output of operation.
    public func fetch(_ operation: String, index: Int = 0) throws -> Runner {
      let op = try g.searchOperation(forName: operation)
      return fetch(op, index: index)
    }//end fetch

    ///  Make run() execute operation, but not return any evaluated Tensors.
    public func addTarget(_ operation: Operation) -> Runner {
      targets.append(operation)
      return self
    }//end addTarget

    ///  Make run() execute operation, but not return any evaluated Tensors.
    public func addTarget(_ operation: String) throws -> Runner {
      let op = try g.searchOperation(forName: operation)
      return self.addTarget(op)
    }//end func

    /// (Experimental method): set options (typically for debugging) for this run.
    public func setOptions(_ options: Buffer) throws -> Runner {
      self.runOptions = options
      return self
    }//end func

    public init(graph: Graph) throws {
      g = graph
      session = try g.newSession()
    }//end init

    public init(graph: Graph, sessionOptions: SessionOptions? = nil, runOptions: Buffer? = nil, exportDir: String, tags: [String], metaGraphDef: Buffer) throws {
      g = graph
      session = try Session(sessionOptions: sessionOptions, runOptions: runOptions, exportDir: exportDir, tags: tags, graph: g, metaGraphDef: metaGraphDef)
    }
    /// Execute the graph fragments necessary to compute all requested fetches.
    public func run() throws -> [Tensor] {
      let meta = try Buffer()
      return try session.run(inputs: inputs, outputs: outputs, targets: targets, options: runOptions, metaData: meta)
    }//end run
  }//end class

  public class PartialSession {

    let handle: UnsafeMutablePointer<CChar>
    let parent: OpaquePointer

    /// Set up the graph with the intended feeds (inputs) and fetches (outputs) for a sequence of partial run calls.
    /// On success, returns a handle that is used for subsequent PRun calls. The handle should be deleted with TF_DeletePRunHandle when it is no longer needed.
    /// On failure, out_status contains a tensorflow::Status with an error message. NOTE: This is EXPERIMENTAL and subject to change.
    /// - parameters:
    ///   - session: the parent session
    ///   - inputs: an array of Output as inputs
    ///   - outputs: an array of Output as outputs
    ///   - targets: target operations in an array
    /// - throws: Panic
    public init(session: Session, inputs: [Output]? = nil, outputs: [Output]? = nil, targets: [Operation]? = nil) throws {

      guard let _ = TFLib.libDLL else { throw Panic.DLL(reason: "Library is Missing") }

      var h = UnsafeMutablePointer<CChar>(bitPattern: 0)
      let s = try Status()
      parent = session.session

      let szInputs: Int
      if let sz = inputs?.count, sz > 0 {
        szInputs = sz
      } else {
        szInputs = 0
      }//end if

      let szOutputs: Int
      if let sz = outputs?.count, sz > 0 {
        szOutputs = sz
      } else {
        szOutputs = 0
      }//end if

      let szTargets: Int
      if let sz = targets?.count, sz > 0 {
        szTargets = sz
      } else {
        szTargets = 0
      }//end if

      let pTargets = targets?.map { $0.operation }

      TFLib.SessionPRunSetup(
        parent,
        inputs?.withUnsafeBufferPointer { $0.baseAddress }, Int32(szInputs),
        outputs?.withUnsafeBufferPointer { $0.baseAddress }, Int32(szOutputs),
        pTargets?.withUnsafeBufferPointer { $0.baseAddress }, Int32(szTargets),
        &h, s.status)

      guard s.code == .OK, let hSession = h else
      { throw Panic.FAULT(reason: s.message) }

      handle = hSession
    }//end init

    deinit {
      TFLib.DeletePRunHandle(handle)
    }//end init

    /// Continue to run the graph with additional feeds and fetches. The execution state is uniquely identified by the handle. NOTE: This is EXPERIMENTAL and subject to change.
    /// - parameters:
    ///   - inputs: a pair array of input / tensor
    ///   - outputs: output array
    ///   - target: target array
    /// - returns: result tensors in an array
    /// - throws: Panic.
    public func run
      (inputs: [(Output, Tensor)]? = nil,
       outputs: [Output]? = nil, targets: [Operation]? = nil
      ) throws -> [Tensor]
    {
      guard let _ = TFLib.libDLL else { throw Panic.DLL(reason: "Lib Not Ready")}

      let inpVar = inputs?.map { $0.0 }
      let inpVal = inputs?.map { $0.1.tensor }
      let pTargets = targets?.map { $0.operation }

      let szInputs: Int
      if let sz = inputs?.count, sz > 0 {
        szInputs = sz
      } else {
        szInputs = 0
      }//end if

      let szOutputs: Int
      if let sz = outputs?.count, sz > 0 {
        szOutputs = sz
      } else {
        szOutputs = 0
      }//end if

      let szTargets: Int
      if let sz = targets?.count, sz > 0 {
        szTargets = sz
      } else {
        szTargets = 0
      }//end if

      let pOutputValues: UnsafeMutablePointer<OpaquePointer>?

      if szOutputs > 0 {
        pOutputValues = UnsafeMutablePointer<OpaquePointer>.allocate(capacity: szOutputs)
      } else {
        pOutputValues = UnsafeMutablePointer<OpaquePointer>(bitPattern: 0)
      }//end if

      let s = try Status()

      TFLib.SessionPRun(
        parent, handle,

        inpVar?.withUnsafeBufferPointer { $0.baseAddress },
        inpVal?.withUnsafeBufferPointer { $0.baseAddress },
        Int32(szInputs),

        outputs?.withUnsafeBufferPointer { $0.baseAddress },
        pOutputValues,
        Int32(szOutputs),

        pTargets?.withUnsafeBufferPointer { $0.baseAddress },
        Int32(szTargets),s.status)

      let outputTensors: [Tensor]

      if szOutputs > 0, let p = pOutputValues {
        let values = UnsafeMutableBufferPointer<OpaquePointer>(start: p, count: szOutputs)
        // WARNING: tensors created here are WILD - autodestroy = false
        outputTensors = Array(values).map { Tensor(handle: $0) }
        p.deallocate(capacity: szOutputs)
      } else {
        outputTensors = [Tensor]()
      }//end if


      guard s.code == .OK else
      { throw Panic.FAULT(reason: s.message) }

      // WARNING: all tensors created by array must set autodestroy to true!!!
      outputTensors.forEach { $0.autoDestroy = true }

      return outputTensors
    }//end func
  }//end class

  /// plug in modules loader
  public class Library {
    let handle: OpaquePointer

    /// load a library by path
    /// - parameters:
    ///   - path: path of the plug in module
    /// - throws: Panic
    public init(path: String) throws {

      let status = try Status()

      guard let _ = TFLib.libDLL else
      { throw Panic.DLL(reason: "Open module before primary library") }

      guard let hLib = TFLib.LoadLibrary(path, status.status) ,
        status.code == .OK else
      { throw Panic.FAULT(reason: status.message) }

      handle = hLib
    }//end init

    deinit {
      TFLib.DeleteLibraryHandle(handle)
    }//end deinit

    /// return all operations in this library as a protocol buffer
    public var operations: Buffer {
      get {
        var buf = TFLib.GetOpList(handle)
        return Buffer(buf: &buf)
      } //end get
    }//end operations
  }//end class

  /// TF_Version returns a string describing version information of the
  /// TensorFlow library. TensorFlow using semantic versioning.
  public static var Version: String {
    get {
      if let _ = TFLib.libDLL, let ver = TFLib.Version() {
        return String(cString: ver)
      }else {
        return ""
      }//end if
    }//end get
  }//end var


}//end class
