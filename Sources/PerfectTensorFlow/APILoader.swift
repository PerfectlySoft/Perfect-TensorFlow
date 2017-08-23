//
//  APILoader.swift
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
import TensorFlowAPI

/// Static C API of TensorFlow
public class TFLib {

  /// Exceptions
  public enum Panic: Error {
    case
    /// Exceptions in dynamic library loading
    DLL(reason: String),
    /// Exceptions in api binding
    SYM(reason: String),
    /// Exceptions in calling
    CALL,
    /// One of the parameters passed to the wrapper is not acceptable
    INVALID,
    /// fault with reason
    FAULT(reason: String)
  }

  /// TF_Code holds an error code.  The enum values here are identical to
  /// corresponding values in error_codes.proto.
  public enum Code: Int {
    /// -1 is added by Perfect
    case UNDEFINED = -1,
    OK = 0,
    CANCELLED = 1,
    UNKNOWN = 2,
    INVALID_ARGUMENT = 3,
    DEADLINE_EXCEEDED = 4,
    NOT_FOUND = 5,
    ALREADY_EXISTS = 6,
    PERMISSION_DENIED = 7,
    UNAUTHENTICATED = 16,
    RESOURCE_EXHAUSTED = 8,
    FAILED_PRECONDITION = 9,
    ABORTED = 10,
    OUT_OF_RANGE = 11,
    UNIMPLEMENTED = 12,
    INTERNAL = 13,
    UNAVAILABLE = 14,
    DATA_LOSS = 15
  }

  public static var libDLL = UnsafeMutableRawPointer(bitPattern: 0)

  public static func LoadFunction <T> (_ library: UnsafeMutableRawPointer?, _ symbol: String) throws -> T {
    guard let lib = library else {
      throw Panic.DLL(reason: String(cString: dlerror()))
    }//end guard
    guard let sym = dlsym(lib, symbol) else {
      throw Panic.SYM(reason: String(cString: dlerror()))
    }//end guard
    return unsafeBitCast(sym, to: T.self)
  }

  public static var Version: @convention(c) () -> UnsafePointer<CChar>? = { return  nil }
  public static var DataTypeSize: @convention(c) (Int32) -> Int = { _ in return 0 }

  /// Return a new status object.
  public static var NewStatus: @convention(c) () -> OpaquePointer? = { return OpaquePointer(bitPattern: 0) }

  /// Delete a previously created status object.
  public static var DeleteStatus: @convention(c) (OpaquePointer) -> Void = { _ in }

  /// Record <code, msg> in *s.  Any previous information is lost.
  /// A common use is to clear a status: TF_SetStatus(s, TF_OK, "");
  public static var SetStatus: @convention(c) (OpaquePointer, Int32, UnsafePointer<CChar>?) -> Void = { _, _ , _ in }

  /// Return the code record in *s.
  public static var GetCode: @convention(c) (OpaquePointer) -> Int32 = { _ in return 0 }

  /// Return a pointer to the (null-terminated) error message in *s.  The
  /// return value points to memory that is only usable until the next
  /// mutation to *s.  Always returns an empty string if TF_GetCode(s) is
  /// TF_OK.
  public static var Message: @convention(c) (OpaquePointer) -> UnsafePointer<CChar>? = { _ in return UnsafePointer(bitPattern: 0) }
  public static var NewBufferFromString: @convention(c) (UnsafePointer<CChar>, Int) -> UnsafeMutablePointer<TF_Buffer>? = { _, _ in nil }
  public static var NewBuffer: @convention(c) () -> UnsafeMutablePointer<TF_Buffer>? = { return nil }
  public static var DeleteBuffer: @convention(c) (UnsafeMutablePointer<TF_Buffer>) -> Void = { _ in }
  /// TF_Tensor holds a multi-dimensional array of elements of a single data type.
  /// For all types other than TF_STRING, the data buffer stores elements
  /// in row major order.  E.g. if data is treated as a vector of TF_DataType:
  ///
  ///   element 0:   index (0, ..., 0)
  ///   element 1:   index (0, ..., 1)
  ///   ...
  ///
  /// The format for TF_STRING tensors is:
  ///   start_offset: array[uint64]
  ///   data:         byte[...]
  ///
  ///   The string length (as a varint), followed by the contents of the string
  ///   is encoded at data[start_offset[i]]]. TF_StringEncode and TF_StringDecode
  ///   facilitate this encoding.

  /// Return a new tensor that holds the bytes data[0,len-1].
  ///
  /// The data will be deallocated by a subsequent call to TF_DeleteTensor via:
  ///      (*deallocator)(data, len, deallocator_arg)
  /// Clients must provide a custom deallocator function so they can pass in
  /// memory managed by something like numpy.
  public static var NewTensor: @convention(c) (Int32, UnsafePointer<Int64>?, Int32, UnsafeMutablePointer<Int8>?, Int, ((UnsafeMutablePointer<Int8>?, Int, OpaquePointer?) -> Void), OpaquePointer?) -> OpaquePointer? = { _, _, _, _, _, _, _  in return nil }

  /// Allocate and return a new Tensor.
  ///
  /// This function is an alternative to TF_NewTensor and should be used when
  /// memory is allocated to pass the Tensor to the C API. The allocated memory
  /// satisfies TensorFlow's memory alignment preferences and should be preferred
  /// over calling malloc and free.
  ///
  /// The caller must set the Tensor values by writing them to the pointer returned
  /// by TF_TensorData with length TF_TensorByteSize.
  public static var AllocateTensor: @convention(c) (Int32, UnsafePointer<Int64>?, Int32, Int) -> OpaquePointer? = { _, _, _, _ in return nil }

  /// Destroy a tensor.
  public static var DeleteTensor: @convention(c) (OpaquePointer) -> Void = { _ in }

  /// Return the type of a tensor element.
  public static var TensorType: @convention(c) (OpaquePointer) -> Int32 = { _ in return 0 }

  /// Return the number of dimensions that the tensor has.
  public static var NumDims: @convention(c) (OpaquePointer) -> Int32 = { _ in return 0 }

  /// Return the length of the tensor in the "dim_index" dimension.
  /// REQUIRES: 0 <= dim_index < TF_NumDims(tensor)
  public static var Dim: @convention(c) (OpaquePointer, Int32) -> Int64 = { _, _ in return 0 }

  /// Return the size of the underlying data in bytes.
  public static var TensorByteSize: @convention(c) (OpaquePointer) -> Int = { _ in return 0 }

  /// Return a pointer to the underlying data buffer.
  public static var TensorData: @convention(c) (OpaquePointer) -> UnsafeMutableRawPointer? = { _ in return nil }

  /// Encode the string `src` (`src_len` bytes long) into `dst` in the format
  /// required by TF_STRING tensors. Does not write to memory more than `dst_len`
  /// bytes beyond `*dst`. `dst_len` should be at least
  /// TF_StringEncodedSize(src_len).
  ///
  /// On success returns the size in bytes of the encoded string.
  /// Returns an error into `status` otherwise.
  public static var StringEncode: @convention(c) (UnsafePointer<CChar>, Int, UnsafeMutablePointer<CChar>, Int, OpaquePointer) -> Int = { _, _, _, _, _ in return 0}

  /// Decode a string encoded using TF_StringEncode.
  ///
  /// On success, sets `*dst` to the start of the decoded string and `*dst_len` to
  /// its length. Returns the number of bytes starting at `src` consumed while
  /// decoding. `*dst` points to memory within the encoded buffer.  On failure,
  /// `*dst` and `*dst_len` are undefined and an error is set in `status`.
  ///
  /// Does not read memory more than `src_len` bytes beyond `src`.
  public static var StringDecode: @convention(c) (UnsafePointer<CChar>, Int, UnsafePointer<UnsafeMutablePointer<CChar>?>, UnsafeMutablePointer<Int>, OpaquePointer) -> Int = { _, _, _, _, _ in return 0 }

  /// Return the size in bytes required to encode a string `len` bytes long into a
  /// TF_STRING tensor.
  public static var StringEncodedSize: @convention(c) (Int) -> Int = { _ in return 0 }

  /// TF_SessionOptions holds options that can be passed during session creation.
  public static var NewSessionOptions: @convention(c) ( ) -> OpaquePointer? = { nil }

  /// Set the target in TF_SessionOptions.options.
  /// target can be empty, a single entry, or a comma separated list of entries.
  /// Each entry is in one of the following formats :
  /// "local"
  /// ip:port
  /// host:port
  public static var SetTarget: @convention(c) (OpaquePointer, UnsafePointer<CChar>) -> Void = { _, _ in }

  /// Set the config in TF_SessionOptions.options.
  /// config should be a serialized tensorflow.ConfigProto proto.
  /// If config was not parsed successfully as a ConfigProto, record the
  /// error information in *status.
  public static var SetConfig: @convention(c) (OpaquePointer, UnsafePointer<CChar>, Int, OpaquePointer) -> Void = { _, _, _, _ in }

  /// Destroy an options object.
  public static var DeleteSessionOptions: @convention(c) (OpaquePointer) -> Void = { _ in }

  /// Return a new graph object.
  public static var NewGraph: @convention(c) ( ) -> OpaquePointer? = { return nil }

  /// Destroy an options object.  Graph will be deleted once no more
  /// TFSession's are referencing it.
  public static var DeleteGraph: @convention(c) (OpaquePointer) -> Void = { _ in }

  /// Sets the shape of the Tensor referenced by `output` in `graph` to
  /// the shape described by `dims` and `num_dims`.
  ///
  /// If the number of dimensions is unknown, `num_dims` must be
  /// set to -1 and dims can be null. If a dimension is unknown,
  /// the corresponding entry in the `dims` array must be -1.
  ///
  /// This does not overwrite the existing shape associated with `output`,
  /// but merges the input shape with the existing shape.  For example,
  /// setting a shape of [-1, 2] with an existing shape [2, -1] would set
  /// a final shape of [2, 2] based on shape merging semantics.
  //
  /// Returns an error into `status` if:
  ///   * `output` is not in `graph`.
  ///   * An invalid shape is being set (e.g., the shape being set
  ///     is incompatible with the existing shape).
  public static var GraphSetTensorShape: @convention(c) (OpaquePointer, TF_Output, UnsafePointer<Int64>, Int32, OpaquePointer) -> Void = { _, _, _, _, _ in }

  /// Returns the number of dimensions of the Tensor referenced by `output`
  /// in `graph`.
  ///
  /// If the number of dimensions in the shape is unknown, returns -1.
  ///
  /// Returns an error into `status` if:
  ///   * `output` is not in `graph`.
  public static var GraphGetTensorNumDims: @convention(c) (OpaquePointer, TF_Output, OpaquePointer) -> Int32 = { _, _, _ in return 0 }

  /// Returns the shape of the Tensor referenced by `output` in `graph`
  /// into `dims`. `dims` must be an array large enough to hold `num_dims`
  /// entries (e.g., the return value of TF_GraphGetTensorNumDims).
  ///
  /// If the number of dimensions in the shape is unknown or the shape is
  /// a scalar, `dims` will remain untouched. Otherwise, each element of
  /// `dims` will be set corresponding to the size of the dimension. An
  /// unknown dimension is represented by `-1`.
  ///
  /// Returns an error into `status` if:
  ///   * `output` is not in `graph`.
  ///   * `num_dims` does not match the actual number of dimensions.
  public static var GraphGetTensorShape: @convention(c) (OpaquePointer, TF_Output, UnsafeMutablePointer<Int64>, Int32, OpaquePointer) -> Void = { _, _, _, _, _ in }

  /// Operation will only be added to *graph when TF_FinishOperation() is
  /// called (assuming TF_FinishOperation() does not return an error).
  /// *graph must not be deleted until after TF_FinishOperation() is
  /// called.
  public static var NewOperation: @convention(c) (OpaquePointer, UnsafePointer<CChar>, UnsafePointer<CChar>) -> OpaquePointer? = { _, _, _ in return nil }

  /// Specify the device for `desc`.  Defaults to empty, meaning unconstrained.
  public static var SetDevice: @convention(c) (OpaquePointer, UnsafePointer<CChar>) -> Void = { _, _ in }

  /// The calls to TF_AddInput and TF_AddInputList must match (in number,
  /// order, and type) the op declaration.  For example, the "Concat" op
  /// has registration:
  ///   REGISTER_OP("Concat")
  ///       .Input("concat_dim: int32")
  ///       .Input("values: N * T")
  ///       .Output("output: T")
  ///       .Attr("N: int >= 2")
  ///       .Attr("T: type");
  /// that defines two inputs, "concat_dim" and "values" (in that order).
  /// You must use TF_AddInput() for the first input (since it takes a
  /// single tensor), and TF_AddInputList() for the second input (since
  /// it takes a list, even if you were to pass a list with a single
  /// tensor), as in:
  ///   TF_OperationDescription* desc = TF_NewOperation(graph, "Concat", "c");
  ///   TF_Output concat_dim_input = {...};
  ///   TF_AddInput(desc, concat_dim_input);
  ///   TF_Output values_inputs[5] = {{...}, ..., {...}};
  ///   TF_AddInputList(desc, values_inputs, 5);
  ///
  /// For inputs that take a single tensor.
  public static var AddInput: @convention(c) (OpaquePointer, TF_Output) -> Void = { _, _ in }

  // For inputs that take a list of tensors.
  // inputs must point to TF_Output[num_inputs].
  public static var AddInputList: @convention(c) (OpaquePointer, UnsafePointer<TF_Output>, Int32) -> Void = { _, _, _ in }

  /// Call once per control input to `desc`.
  public static var AddControlInput: @convention(c) (OpaquePointer, OpaquePointer) -> Void = { _, _ in }

  /// Request that `desc` be co-located on the device where `op`
  /// is placed.
  ///
  /// Use of this is discouraged since the implementation of device placement is
  /// subject to change. Primarily intended for public libraries
  public static var ColocateWith: @convention(c) (OpaquePointer, OpaquePointer) -> Void = { _, _ in }

  /// Call some TF_SetAttr*() function for every attr that is not
  /// inferred from an input and doesn't have a default value you wish to
  /// keep.
  ///
  /// `value` must point to a string of length `length` bytes.
  public static var SetAttrString: @convention(c) (OpaquePointer, UnsafePointer<CChar>, UnsafePointer<CChar>, Int) -> Void = { _, _, _, _ in }

  /// `values` and `lengths` each must have lengths `num_values`.
  /// `values[i]` must point to a string of length `lengths[i]` bytes.
  public static var SetAttrStringList: @convention(c) (OpaquePointer, UnsafePointer<CChar>, UnsafePointer<UnsafePointer<CChar>>?, UnsafePointer<Int>?, Int32) -> Void = { _, _, _, _, _ in }
  public static var SetAttrInt: @convention(c) (OpaquePointer, UnsafePointer<CChar>, Int64) -> Void = { _, _, _ in }
  public static var SetAttrIntList: @convention(c) (OpaquePointer, UnsafePointer<CChar>, UnsafePointer<Int64>, Int32) -> Void = { _, _, _, _ in }
  public static var SetAttrFloat: @convention(c) (OpaquePointer, UnsafePointer<CChar>, float_t) -> Void = { _, _, _ in }
  public static var SetAttrFloatList: @convention(c) (OpaquePointer, UnsafePointer<CChar>, UnsafePointer<float_t>, Int32) -> Void = { _, _, _, _ in }
  public static var SetAttrBool: @convention(c) (OpaquePointer, UnsafePointer<CChar>, UInt8) -> Void = { _, _, _ in }
  public static var SetAttrBoolList: @convention(c) (OpaquePointer, UnsafePointer<CChar>, UnsafePointer<UInt8>, Int32) -> Void = { _, _, _, _ in }
  public static var SetAttrType: @convention(c) (OpaquePointer, UnsafePointer<CChar>, Int32) -> Void = { _, _, _ in }
  public static var SetAttrTypeList: @convention(c) (OpaquePointer, UnsafePointer<CChar>, UnsafePointer<Int32>, Int32) -> Void = { _, _, _, _ in }

  /// Set `num_dims` to -1 to represent "unknown rank".  Otherwise,
  /// `dims` points to an array of length `num_dims`.  `dims[i]` must be
  /// >= -1, with -1 meaning "unknown dimension".
  public static var SetAttrShape: @convention(c) (OpaquePointer, UnsafePointer<CChar>, UnsafePointer<Int64>, Int32) -> Void = { _ , _ ,  _,  _ in }

  /// `dims` and `num_dims` must point to arrays of length `num_shapes`.
  /// Set `num_dims[i]` to -1 to represent "unknown rank".  Otherwise,
  /// `dims[i]` points to an array of length `num_dims[i]`.  `dims[i][j]`
  // must be >= -1, with -1 meaning "unknown dimension".
  public static var SetAttrShapeList: @convention(c) (OpaquePointer, UnsafePointer<CChar>, UnsafeMutablePointer<UnsafePointer<Int64>>, UnsafeMutablePointer<Int32>, Int32) -> Void = { _, _, _, _, _ in }

  /// `proto` must point to an array of `proto_len` bytes representing a
  /// binary-serialized TensorShapeProto.
  public static var SetAttrTensorShapeProto: @convention(c) (OpaquePointer, UnsafePointer<CChar>, UnsafePointer<CChar>, Int, OpaquePointer) -> Void = { _, _, _, _, _ in }

  /// `protos` and `proto_lens` must point to arrays of length `num_shapes`.
  /// `protos[i]` must point to an array of `proto_lens[i]` bytes
  /// representing a binary-serialized TensorShapeProto.
  public static var SetAttrTensorShapeProtoList: @convention(c) (OpaquePointer, UnsafePointer<CChar>, UnsafePointer<UnsafePointer<CChar>>, UnsafePointer<Int>, Int32, OpaquePointer) -> Void = { _, _, _, _, _, _ in }

  public static var SetAttrTensor: @convention(c) (OpaquePointer, UnsafePointer<CChar>, OpaquePointer, OpaquePointer) -> Void = { _, _, _, _ in }

  public static var SetAttrTensorList: @convention(c) (OpaquePointer, UnsafePointer<CChar>, UnsafePointer<OpaquePointer>, Int32, OpaquePointer) -> Void = { _, _, _, _, _ in }

  public static var SetAttrValueProto: @convention(c) (OpaquePointer, UnsafePointer<CChar>, UnsafePointer<CChar>, Int, OpaquePointer) -> Void = { _, _, _, _, _ in }

  /// If this function succeeds:
  ///   * *status is set to an OK value,
  ///   * a TF_Operation is added to the graph,
  ///   * a non-null value pointing to the added operation is returned --
  ///     this value is valid until the underlying graph is deleted.
  /// Otherwise:
  ///   * *status is set to a non-OK value,
  ///   * the graph is not modified,
  ///   * a null value is returned.
  /// In either case, it deletes `desc`.
  public static var FinishOperation: @convention(c) (OpaquePointer, OpaquePointer) -> OpaquePointer = { op, _ in return op }

  /// TF_Operation functions.  Operations are immutable once created, so
  /// these are all query functions.
  public static var OperationName: @convention(c) (OpaquePointer) -> UnsafePointer<CChar>? = { _ in return nil }
  public static var OperationOpType: @convention(c) (OpaquePointer) -> UnsafePointer<CChar>? = { _ in return nil }
  public static var OperationDevice: @convention(c) (OpaquePointer) -> UnsafePointer<CChar>? = { _ in return nil }

  public static var OperationNumOutputs: @convention(c) (OpaquePointer) -> Int32 = { _ in return 0 }
  public static var OperationOutputType: @convention(c) (TF_Output) -> Int32 = { _ in return 0 }
  public static var OperationOutputListLength: @convention(c) (OpaquePointer, UnsafePointer<CChar>, OpaquePointer) -> Int32 = { _, _, _ in return 0 }

  public static var OperationNumInputs: @convention(c) (OpaquePointer) -> Int32 = { _ in return 0 }
  public static var OperationInputType: @convention(c) (TF_Input) -> Int32 = { _ in return 0 }
  public static var OperationInputListLength: @convention(c) (OpaquePointer, UnsafePointer<CChar>, OpaquePointer) -> Int32 = { _, _, _ in return 0 }

  /// In this code:
  ///   TF_Output producer = TF_OperationInput(consumer);
  /// There is an edge from producer.oper's output (given by
  /// producer.index) to consumer.oper's input (given by consumer.index).
  public static var OperationInput: @convention(c) (TF_Input) -> TF_Output = { _ in TF_Output() }

  /// Get the number of current consumers of a specific output of an
  /// operation.  Note that this number can change when new operations
  /// are added to the graph.
  public static var OperationOutputNumConsumers: @convention(c) (TF_Output) -> Int32 = { _ in return 0 }

  /// Get list of all current consumers of a specific output of an
  /// operation.  `consumers` must point to an array of length at least
  /// `max_consumers` (ideally set to
  /// TF_OperationOutputNumConsumers(oper_out)).  Beware that a concurrent
  /// modification of the graph can increase the number of consumers of
  /// an operation.  Returns the number of output consumers (should match
  /// TF_OperationOutputNumConsumers(oper_out)).
  public static var OperationOutputConsumers: @convention(c) (TF_Output, UnsafeMutablePointer<TF_Input>, Int32) -> Int32 = { _, _, _ in return 0 }

  /// Get the number of control inputs to an operation.
  public static var OperationNumControlInputs: @convention(c) (OpaquePointer) -> Int32 = { _ in return 0 }

  /// Get list of all control inputs to an operation.  `control_inputs` must
  /// point to an array of length `max_control_inputs` (ideally set to
  /// TF_OperationNumControlInputs(oper)).  Returns the number of control
  /// inputs (should match TF_OperationNumControlInputs(oper)).
  public static var OperationGetControlInputs: @convention(c) (OpaquePointer, UnsafeMutablePointer<OpaquePointer>, Int32) -> Int32 = { _, _, _ in return 0 }

  /// Get the number of operations that have `*oper` as a control input.
  /// Note that this number can change when new operations are added to
  /// the graph.
  public static var OperationNumControlOutputs: @convention(c) (OpaquePointer) -> Int32 = { _ in return 0 }

  /// Get the list of operations that have `*oper` as a control input.
  /// `control_outputs` must point to an array of length at least
  /// `max_control_outputs` (ideally set to
  /// TF_OperationNumControlOutputs(oper)). Beware that a concurrent
  /// modification of the graph can increase the number of control
  /// outputs.  Returns the number of control outputs (should match
  /// TF_OperationNumControlOutputs(oper)).
  public static var OperationGetControlOutputs: @convention(c) (OpaquePointer, UnsafeMutablePointer<OpaquePointer>, Int32) -> Int32 = { _, _, _ in return 0 }

  /// Returns metadata about the value of the attribute `attr_name` of `oper`.
  public static var OperationGetAttrMetadata: @convention(c) (OpaquePointer, UnsafePointer<CChar>, OpaquePointer) -> TF_AttrMetadata = { _, _, _ in return TF_AttrMetadata() }

  /// Fills in `value` with the value of the attribute `attr_name`.  `value` must
  /// point to an array of length at least `max_length` (ideally set to
  /// TF_AttrMetadata.total_size from TF_OperationGetAttrMetadata(oper,
  /// attr_name)).
  public static var OperationGetAttrString: @convention(c) (OpaquePointer, UnsafePointer<CChar>, UnsafeMutableRawPointer, Int, OpaquePointer) -> Void = { _, _, _, _, _ in }

  /// Get the list of strings in the value of the attribute `attr_name`.  Fills in
  /// `values` and `lengths`, each of which must point to an array of length at
  /// least `max_values`.
  ///
  /// The elements of values will point to addresses in `storage` which must be at
  /// least `storage_size` bytes in length.  Ideally, max_values would be set to
  /// TF_AttrMetadata.list_size and `storage` would be at least
  /// TF_AttrMetadata.total_size, obtained from TF_OperationGetAttrMetadata(oper,
  /// attr_name).
  ///
  /// Fails if storage_size is too small to hold the requested number of strings.
  public static var OperationGetAttrStringList: @convention(c) (OpaquePointer, UnsafePointer<CChar>, UnsafeMutablePointer<UnsafeMutablePointer<CChar>>, UnsafeMutablePointer<Int>, Int32, UnsafeMutablePointer<CChar>, Int, OpaquePointer) -> Void = { _, _, _, _, _, _, _, _ in }

  public static var OperationGetAttrInt: @convention(c) (OpaquePointer, UnsafePointer<CChar>, UnsafeMutablePointer<Int64>, OpaquePointer) -> Void = { _, _, _, _ in }

  // Fills in `values` with the value of the attribute `attr_name` of `oper`.
  // `values` must point to an array of length at least `max_values` (ideally set
  // TF_AttrMetadata.list_size from TF_OperationGetAttrMetadata(oper,
  // attr_name)).

  public static var OperationGetAttrIntList: @convention(c) (OpaquePointer, UnsafePointer<CChar>, UnsafeMutablePointer<Int64>, Int32, OpaquePointer) -> Void = { _, _ , _, _, _ in }

  /// Fills in `values` with the value of the attribute `attr_name` of `oper`.
  /// `values` must point to an array of length at least `max_values` (ideally set
  /// TF_AttrMetadata.list_size from TF_OperationGetAttrMetadata(oper,
  /// attr_name)).
  public static var OperationGetAttrFloat: @convention(c) (OpaquePointer, UnsafePointer<CChar>, UnsafeMutablePointer<float_t>, OpaquePointer) -> Void = { _, _, _, _ in }

  //// Fills in `values` with the value of the attribute `attr_name` of `oper`.
  /// `values` must point to an array of length at least `max_values` (ideally set
  /// to TF_AttrMetadata.list_size from TF_OperationGetAttrMetadata(oper,
  /// attr_name)).
  public static var OperationGetAttrFloatList: @convention(c) (OpaquePointer, UnsafePointer<CChar>, UnsafeMutablePointer<float_t>, Int32, OpaquePointer) -> Void = { _, _, _, _, _ in }

  public static var OperationGetAttrBool: @convention(c) (OpaquePointer, UnsafePointer<CChar>, UnsafeMutablePointer<UInt8>, OpaquePointer) -> Void = { _, _, _, _ in }

  /// Fills in `values` with the value of the attribute `attr_name` of `oper`.
  /// `values` must point to an array of length at least `max_values` (ideally set
  /// to TF_AttrMetadata.list_size from TF_OperationGetAttrMetadata(oper,
  /// attr_name)).
  public static var OperationGetAttrBoolList: @convention(c) (OpaquePointer, UnsafePointer<CChar>, UnsafeMutablePointer<UInt8>, Int32, OpaquePointer) -> Void = { _, _, _, _, _ in }

  public static var OperationGetAttrType: @convention(c) (OpaquePointer, UnsafePointer<CChar>, UnsafeMutablePointer<Int32>, OpaquePointer) -> Void = { _, _, _, _ in }

  /// Fills in `values` with the value of the attribute `attr_name` of `oper`.
  /// `values` must point to an array of length at least `max_values` (ideally set
  /// to TF_AttrMetadata.list_size from TF_OperationGetAttrMetadata(oper,
  /// attr_name)).
  public static var OperationGetAttrTypeList: @convention(c) (OpaquePointer, UnsafePointer<CChar>, UnsafeMutablePointer<Int32>, Int32, OpaquePointer) -> Void = { _, _, _, _, _ in }

  /// Fills in `value` with the value of the attribute `attr_name` of `oper`.
  /// `values` must point to an array of length at least `num_dims` (ideally set to
  /// TF_Attr_Meta.size from TF_OperationGetAttrMetadata(oper, attr_name)).
  public static var OperationGetAttrShape: @convention(c) (OpaquePointer, UnsafePointer<CChar>, UnsafeMutablePointer<Int64>, Int32, OpaquePointer) -> Void = { _, _, _, _, _ in }

  /// Fills in `dims` with the list of shapes in the attribute `attr_name` of
  /// `oper` and `num_dims` with the corresponding number of dimensions. On return,
  /// for every i where `num_dims[i]` > 0, `dims[i]` will be an array of
  /// `num_dims[i]` elements. A value of -1 for `num_dims[i]` indicates that the
  /// i-th shape in the list is unknown.
  ///
  /// The elements of `dims` will point to addresses in `storage` which must be
  /// large enough to hold at least `storage_size` int64_ts.  Ideally, `num_shapes`
  /// would be set to TF_AttrMetadata.list_size and `storage_size` would be set to
  /// TF_AttrMetadata.total_size from TF_OperationGetAttrMetadata(oper,
  /// attr_name).
  ///
  /// Fails if storage_size is insufficient to hold the requested shapes.
  public static var OperationGetAttrShapeList: @convention(c) (OpaquePointer, UnsafePointer<CChar>, UnsafeMutablePointer<UnsafeMutablePointer<Int64>>, UnsafeMutablePointer<Int32>, Int32, UnsafeMutablePointer<Int64>, Int32, OpaquePointer) -> Void = { _, _, _, _, _, _, _, _ in }

  /// Sets `value` to the binary-serialized TensorShapeProto of the value of
  /// `attr_name` attribute of `oper`'.
  public static var OperationGetAttrTensorShapeProto: @convention(c) (OpaquePointer, UnsafePointer<CChar>, UnsafeMutablePointer<TF_Buffer>, OpaquePointer) -> Void = { _, _, _, _ in }

  /// Fills in `values` with binary-serialized TensorShapeProto values of the
  /// attribute `attr_name` of `oper`. `values` must point to an array of length at
  /// least `num_values` (ideally set to TF_AttrMetadata.list_size from
  /// TF_OperationGetAttrMetadata(oper, attr_name)).
  public static var OperationGetAttrTensorShapeProtoList: @convention(c) (OpaquePointer, UnsafePointer<CChar>, UnsafeMutablePointer<UnsafeMutablePointer<TF_Buffer>>, Int32, OpaquePointer) -> Void = { _, _, _, _, _ in }

  /// Gets the TF_Tensor valued attribute of `attr_name` of `oper`.
  ///
  /// Allocates a new TF_Tensor which the caller is expected to take
  /// ownership of (and can deallocate using TF_DeleteTensor).
  public static var OperationGetAttrTensor: @convention(c) (OpaquePointer, UnsafePointer<CChar>, UnsafeMutablePointer<OpaquePointer?>, OpaquePointer) -> Void = { _, _, _, _ in }

  /// Fills in `values` with the TF_Tensor values of the attribute `attr_name` of
  /// `oper`. `values` must point to an array of TF_Tensor* of length at least
  /// `max_values` (ideally set to TF_AttrMetadata.list_size from
  /// TF_OperationGetAttrMetadata(oper, attr_name)).
  ///
  /// The caller takes ownership of all the non-null TF_Tensor* entries in `values`
  /// (which can be deleted using TF_DeleteTensor(values[i])).
  public static var OperationGetAttrTensorList: @convention(c) (OpaquePointer, UnsafePointer<CChar>, UnsafeMutablePointer<OpaquePointer>, Int32, OpaquePointer) -> Void = { _, _, _, _, _ in }

  /// Sets `output_attr_value` to the binary-serialized AttrValue proto
  /// representation of the value of the `attr_name` attr of `oper`.
  public static var OperationGetAttrValueProto: @convention(c) (OpaquePointer, UnsafePointer<CChar>, UnsafeMutablePointer<TF_Buffer>, OpaquePointer) -> Void = { _, _, _, _ in }

  /// Returns the operation in the graph with `oper_name`. Returns nullptr if
  /// no operation found.
  public static var GraphOperationByName: @convention(c) (OpaquePointer, UnsafePointer<CChar>) -> OpaquePointer? = { _, _ in return nil }

  /// Iterate through the operations of a graph.  To use:
  /// size_t pos = 0;
  /// TF_Operation* oper;
  /// while ((oper = TF_GraphNextOperation(graph, &pos)) != nullptr) {
  ///   DoSomethingWithOperation(oper);
  /// }
  public static var GraphNextOperation: @convention(c) (OpaquePointer, UnsafeMutablePointer<Int>) -> OpaquePointer? = { _, _ in return nil }

  /// Write out a serialized representation of `graph` (as a GraphDef protocol
  /// message) to `output_graph_def` (allocated by TF_NewBuffer()).
  /// `output_graph_def`'s underlying buffer will be freed when TF_DeleteBuffer()
  /// is called.
  ///
  /// May fail on very large graphs in the future.
  public static var GraphToGraphDef: @convention(c) (OpaquePointer, UnsafeMutablePointer<TF_Buffer>, OpaquePointer) -> Void = { _, _, _ in }

  /// TF_ImportGraphDefOptions holds options that can be passed to
  /// TF_GraphImportGraphDef.
  public static var NewImportGraphDefOptions: @convention(c) () -> OpaquePointer? = { return nil }
  public static var DeleteImportGraphDefOptions: @convention(c) (OpaquePointer) -> Void = { _ in }

  /// Set the prefix to be prepended to the names of nodes in `graph_def` that will
  /// be imported into `graph`.
  public static var ImportGraphDefOptionsSetPrefix: @convention(c) (OpaquePointer, UnsafePointer<CChar>) -> Void = { _, _ in }

  /// Set any imported nodes with input `src_name:src_index` to have that input
  /// replaced with `dst`. `src_name` refers to a node in the graph to be imported,
  /// `dst` references a node already existing in the graph being imported into.
  public static var ImportGraphDefOptionsAddInputMapping: @convention(c) (OpaquePointer, UnsafePointer<CChar>, Int32, TF_Output) -> Void = { _, _, _, _ in }

  /// Set any imported nodes with control input `src_name` to have that input
  /// replaced with `dst`. `src_name` refers to a node in the graph to be imported,
  /// `dst` references an operation already existing in the graph being imported
  /// into.
  public static var ImportGraphDefOptionsRemapControlDependency: @convention(c) (OpaquePointer, UnsafePointer<CChar>, OpaquePointer) -> Void = { _, _, _ in }

  /// Cause the imported graph to have a control dependency on `oper`. `oper`
  /// should exist in the graph being imported into.
  public static var ImportGraphDefOptionsAddControlDependency: @convention(c) (OpaquePointer, OpaquePointer) -> Void = { _, _ in }

  /// Add an output in `graph_def` to be returned via the `return_outputs` output
  /// parameter of TF_GraphImportGraphDef(). If the output is remapped via an input
  /// mapping, the corresponding existing tensor in `graph` will be returned.
  public static var ImportGraphDefOptionsAddReturnOutput: @convention(c) (OpaquePointer, UnsafePointer<CChar>, Int32) -> Void = { _, _, _ in }

  /// Returns the number of return outputs added via
  /// TF_ImportGraphDefOptionsAddReturnOutput().
  public static var ImportGraphDefOptionsNumReturnOutputs: @convention(c) (OpaquePointer) -> Int32 = { _ in return 0 }

  /// Import the graph serialized in `graph_def` into `graph`.
  ///
  /// `num_return_outputs` must be the number of return outputs added (i.e. the
  /// result of TF_ImportGraphDefOptionsNumReturnOutputs()).  If
  /// `num_return_outputs` is non-zero, `return_outputs` must be of length
  /// `num_return_outputs`. Otherwise it can be null.
  public static var GraphImportGraphDefWithReturnOutputs: @convention(c) (OpaquePointer, UnsafePointer<TF_Buffer>, OpaquePointer, UnsafeMutablePointer<TF_Output>, Int32, OpaquePointer) -> Void = { _, _, _, _, _, _ in }

  /// Import the graph serialized in `graph_def` into `graph`.
  /// Convenience function for when no return outputs have been added.
  public static var GraphImportGraphDef: @convention(c) (OpaquePointer, UnsafePointer<TF_Buffer>, OpaquePointer, OpaquePointer) -> Void = { _, _, _, _ in }

  /// Note: The following function may fail on very large protos in the future.
  public static var OperationToNodeDef: @convention(c) (OpaquePointer, UnsafeMutablePointer<TF_Buffer>, OpaquePointer) -> Void = { _, _, _ in }


  /// Creates a TF_WhileParams for creating a while loop in `g`. `inputs` are
  /// outputs that already exist in `g` used as initial values for the loop
  /// variables.
  ///
  /// The returned TF_WhileParams will have all fields initialized except
  /// `cond_output`, `body_outputs`, and `name`. The `body_outputs` buffer will be
  /// allocated to size `ninputs`. The caller should build `cond_graph` and
  /// `body_graph` starting from the inputs, and store the final outputs in
  /// `cond_output` and `body_outputs`.
  ///
  /// If `status` is OK, the caller must call either TF_FinishWhile or
  /// TF_AbortWhile on the returned TF_WhileParams. If `status` isn't OK, the
  /// returned TF_WhileParams is not valid, and the caller should not call
  /// TF_FinishWhile() or TF_AbortWhile().
  ///
  /// Missing functionality (TODO):
  /// - Gradients (not yet implmented for any ops)
  /// - Reference-type inputs
  /// - Directly referencing external tensors from the cond/body graphs (this is
  ///   possible in the Python API)
  public static var NewWhile: @convention(c) (OpaquePointer, UnsafePointer<TF_Output>, Int32, OpaquePointer) -> TF_WhileParams = { _, _, _, _ in return TF_WhileParams() }

  /// Builds the while loop specified by `params` and returns the output tensors of
  /// the while loop in `outputs`. `outputs` should be allocated to size
  /// `params.ninputs`.
  ///
  /// `params` is no longer valid once this returns.
  ///
  /// Either this or TF_AbortWhile() must be called after a successful
  /// TF_NewWhile() call.
  public static var FinishWhile: @convention(c) (UnsafePointer<TF_WhileParams>, OpaquePointer, UnsafeMutablePointer<TF_Output>) -> Void = { _, _, _ in }

  /// Frees `params`s resources without building a while loop. `params` is no
  /// longer valid after this returns. Either this or TF_FinishWhile() must be
  /// called after a successful TF_NewWhile() call.
  public static var AbortWhile: @convention(c) (UnsafePointer<TF_WhileParams>) -> Void = { _ in }

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
  /// WARNING: This function does not yet support all the gradients that python
  /// supports. See
  /// https://www.tensorflow.org/code/tensorflow/cc/gradients/README.md
  /// for instructions on how to add C++ more gradients.
  public static var AddGradients: @convention(c) (OpaquePointer, UnsafePointer<TF_Output>?, Int32, UnsafePointer<TF_Output>?, Int32, UnsafePointer<TF_Output>?, OpaquePointer, UnsafeMutablePointer<TF_Output>?) -> Void = { _, _, _, _, _, _, _, _  in }


  /// Return a new execution session with the associated graph, or NULL on error.
  ///
  /// *graph must be a valid graph (not deleted or nullptr).  This function will
  /// prevent the graph from being deleted until TF_DeleteSession() is called.
  /// Does not take ownership of opts.
  public static var NewSession: @convention(c) (OpaquePointer, OpaquePointer, OpaquePointer) -> OpaquePointer? = { _, _, _ in return nil }

  /// This function creates a new TF_Session (which is created on success) using
  /// `session_options`, and then initializes state (restoring tensors and other
  /// assets) using `run_options`.
  ///
  /// Any NULL and non-NULL value combinations for (`run_options, `meta_graph_def`)
  /// are valid.
  ///
  /// - `export_dir` must be set to the path of the exported SavedModel.
  /// - `tags` must include the set of tags used to identify one MetaGraphDef in
  ///    the SavedModel.
  /// - `graph` must be a graph newly allocated with TF_NewGraph().
  ///
  /// If successful, populates `graph` with the contents of the Graph and
  /// `meta_graph_def` with the MetaGraphDef of the loaded model.
  public static var LoadSessionFromSavedModel: @convention(c) (OpaquePointer, UnsafePointer<TF_Buffer>, UnsafePointer<CChar>, UnsafePointer<UnsafeMutablePointer<CChar>?>, Int32, OpaquePointer, UnsafeMutablePointer<TF_Buffer>, OpaquePointer) -> OpaquePointer? = { _, _, _, _, _, _, _, _ in return nil }

  /// Close a session.
  ///
  /// Contacts any other processes associated with the session, if applicable.
  /// May not be called after TF_DeleteSession().
  public static var CloseSession: @convention(c) (OpaquePointer, OpaquePointer) -> Void = { _, _ in }

  /// Destroy a session object.
  ///
  /// Even if error information is recorded in *status, this call discards all
  /// local resources associated with the session.  The session may not be used
  /// during or after this call (and the session drops its reference to the
  /// corresponding graph).
  public static var DeleteSession: @convention(c) (OpaquePointer, OpaquePointer) -> Void = { _, _ in }

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
  public static var SessionRun: @convention(c) (OpaquePointer, UnsafePointer<TF_Buffer>?, UnsafePointer<TF_Output>?, UnsafePointer<OpaquePointer>?, Int32, UnsafePointer<TF_Output>?, UnsafeMutablePointer<OpaquePointer>?, Int32, UnsafePointer<OpaquePointer>?, Int32, UnsafeMutablePointer<TF_Buffer>?, OpaquePointer) -> Void = { _, _, _, _, _, _, _, _, _, _, _, _ in }

  /// Lists all devices in a TF_Session.
  ///
  /// Caller takes ownership of the returned TF_DeviceList* which must eventually
  /// be freed with a call to TF_DeleteDeviceList.
  public static var SessionListDevices: @convention(c) (OpaquePointer, OpaquePointer) -> OpaquePointer? = { _, _ in return nil }

  /// Deallocates the device list.
  public static var DeleteDeviceList: @convention(c) (OpaquePointer) -> Void = { _ in}

  /// Counts the number of elements in the device list.
  public static var DeviceListCount: @convention(c) (OpaquePointer) -> Int32 = { _ in return 0 }

  /// Retrieves the full name of the device (e.g. /job:worker/replica:0/...)
  /// The return value will be a pointer to a null terminated string. The caller
  /// must not modify or delete the string. It will be deallocated upon a call to
  /// TF_DeleteDeviceList.
  ///
  /// If index is out of bounds, an error code will be set in the status object,
  /// and a null pointer will be returned.
  public static var DeviceListName: @convention(c) (OpaquePointer, Int32, OpaquePointer) -> UnsafePointer<CChar>? = { _, _, _ in return nil }

  /// Retrieves the type of the device at the given index.
  ///
  /// The caller must not modify or delete the string. It will be deallocated upon
  /// a call to TF_DeleteDeviceList.
  ///
  /// If index is out of bounds, an error code will be set in the status object,
  /// and a null pointer will be returned.
  public static var DeviceListType: @convention(c) (OpaquePointer, Int32, OpaquePointer) -> UnsafePointer<CChar>? = { _, _, _ in return nil }

  /// Retrieve the amount of memory associated with a given device.
  ///
  /// If index is out of bounds, an error code will be set in the status object,
  /// and -1 will be returned.
  public static var DeviceListMemoryBytes: @convention(c) (OpaquePointer, Int32, OpaquePointer) -> Int64 = { _, _, _ in return 0 }

  /// Set up the graph with the intended feeds (inputs) and fetches (outputs) for a
  /// sequence of partial run calls.
  ///
  /// On success, returns a handle that is used for subsequent PRun calls. The
  /// handle should be deleted with TF_DeletePRunHandle when it is no longer
  /// needed.
  ///
  /// On failure, out_status contains a tensorflow::Status with an error
  /// message.
  /// NOTE: This is EXPERIMENTAL and subject to change.
  public static var SessionPRunSetup: @convention(c) (OpaquePointer, UnsafePointer<TF_Output>?, Int32, UnsafePointer<TF_Output>?, Int32, UnsafePointer<OpaquePointer>?, Int32, UnsafePointer<UnsafeMutablePointer<CChar>?>, OpaquePointer) -> Void = { _, _, _, _, _, _, _, _, _ in }

  /// Continue to run the graph with additional feeds and fetches. The
  /// execution state is uniquely identified by the handle.
  /// NOTE: This is EXPERIMENTAL and subject to change.
  public static var SessionPRun: @convention(c) (OpaquePointer, UnsafeMutablePointer<CChar>, UnsafePointer<TF_Output>?, UnsafePointer<OpaquePointer>?, Int32, UnsafePointer<TF_Output>?, UnsafeMutablePointer<OpaquePointer>?, Int32, UnsafePointer<OpaquePointer>?, Int32, OpaquePointer) -> Void = { _, _, _, _, _, _, _, _, _, _, _ in }

  /// Deletes a handle allocated by TF_SessionPRunSetup.
  /// Once called, no more calls to TF_SessionPRun should be made.
  public static var DeletePRunHandle: @convention(c) (UnsafePointer<CChar>) -> Void = { _ in }


  /// Load the library specified by library_filename and register the ops and
  /// kernels present in that library.
  ///
  /// Pass "library_filename" to a platform-specific mechanism for dynamically
  /// loading a library. The rules for determining the exact location of the
  /// library are platform-specific and are not documented here.
  ///
  /// On success, place OK in status and return the newly created library handle.
  /// The caller owns the library handle.
  ///
  /// On failure, place an error status in status and return NULL.
  public static var LoadLibrary: @convention(c) (UnsafePointer<CChar>, OpaquePointer) -> OpaquePointer? = { _, _ in return nil }

  /// Get the OpList of OpDefs defined in the library pointed by lib_handle.
  ///
  /// Returns a TF_Buffer. The memory pointed to by the result is owned by
  /// lib_handle. The data in the buffer will be the serialized OpList proto for
  /// ops defined in the library.
  public static var GetOpList: @convention(c) (OpaquePointer) -> TF_Buffer = { _ in TF_Buffer() }

  /// Frees the memory associated with the library handle.
  /// Does NOT unload the library.
  public static var DeleteLibraryHandle: @convention(c) (OpaquePointer) -> Void = { _ in }

  /// Get the OpList of all OpDefs defined in this address space.
  /// Returns a TF_Buffer, ownership of which is transferred to the caller
  /// (and can be freed using TF_DeleteBuffer).
  ///
  /// The data in the buffer will be the serialized OpList proto for ops registered
  /// in this address space.
  public static var GetAllOpList: @convention(c) () -> UnsafeMutablePointer<TF_Buffer>? = { return nil }

  /// Bootstrap of tensorflow library open, **MUST BE CALL BEFORE ANY OPERATIONS**
  /// - parameters
  ///   - library: the installation path of library TensorFlow for C, /usr/local/lib/libtensorflow.so by default
  /// - throws: Panic
  public static func Open (_ library: String = "/usr/local/lib/libtensorflow.so") throws {
    guard let lib = dlopen(library, RTLD_NOW) else {
      throw Panic.DLL(reason: String(cString: dlerror()))
    }//end lib
    libDLL = lib
    SessionListDevices = try LoadFunction(lib, "TF_SessionListDevices")
    DeleteDeviceList = try LoadFunction(lib, "TF_DeleteDeviceList")
    DeviceListCount = try LoadFunction(lib, "TF_DeviceListCount")
    DeviceListName = try LoadFunction(lib, "TF_DeviceListName")
    DeviceListType = try LoadFunction(lib, "TF_DeviceListType")
    DeviceListMemoryBytes = try LoadFunction(lib, "TF_DeviceListMemoryBytes")
    AddGradients = try LoadFunction(lib, "TF_AddGradients")
    SetAttrValueProto = try LoadFunction(lib, "TF_SetAttrValueProto")
    GetAllOpList = try LoadFunction(lib, "TF_GetAllOpList")
    DeleteLibraryHandle = try LoadFunction(lib, "TF_DeleteLibraryHandle")
    GetOpList = try LoadFunction(lib, "TF_GetOpList")
    LoadLibrary = try LoadFunction(lib, "TF_LoadLibrary")
    DeletePRunHandle = try LoadFunction(lib, "TF_DeletePRunHandle")
    SessionPRun = try LoadFunction(lib, "TF_SessionPRun")
    SessionPRunSetup = try LoadFunction(lib, "TF_SessionPRunSetup")
    SessionRun = try LoadFunction(lib, "TF_SessionRun")
    DeleteSession = try LoadFunction(lib, "TF_DeleteSession")
    CloseSession = try LoadFunction(lib, "TF_CloseSession")
    LoadSessionFromSavedModel = try LoadFunction(lib, "TF_LoadSessionFromSavedModel")
    NewSession = try LoadFunction(lib, "TF_NewSession")
    AbortWhile = try LoadFunction(lib, "TF_AbortWhile")
    FinishWhile = try LoadFunction(lib, "TF_FinishWhile")
    NewWhile = try LoadFunction(lib, "TF_NewWhile")
    OperationToNodeDef = try LoadFunction(lib, "TF_OperationToNodeDef")
    GraphImportGraphDef = try LoadFunction(lib, "TF_GraphImportGraphDef")
    GraphImportGraphDefWithReturnOutputs = try LoadFunction(lib, "TF_GraphImportGraphDefWithReturnOutputs")
    ImportGraphDefOptionsNumReturnOutputs = try LoadFunction(lib, "TF_ImportGraphDefOptionsNumReturnOutputs")
    ImportGraphDefOptionsAddReturnOutput = try LoadFunction(lib, "TF_ImportGraphDefOptionsAddReturnOutput")
    ImportGraphDefOptionsAddControlDependency = try LoadFunction(lib, "TF_ImportGraphDefOptionsAddControlDependency")
    ImportGraphDefOptionsRemapControlDependency = try LoadFunction(lib, "TF_ImportGraphDefOptionsRemapControlDependency")
    ImportGraphDefOptionsAddInputMapping = try LoadFunction(lib, "TF_ImportGraphDefOptionsAddInputMapping")
    ImportGraphDefOptionsSetPrefix = try LoadFunction(lib, "TF_ImportGraphDefOptionsSetPrefix")
    DeleteImportGraphDefOptions = try LoadFunction(lib, "TF_DeleteImportGraphDefOptions")
    NewImportGraphDefOptions = try LoadFunction(lib, "TF_NewImportGraphDefOptions")
    GraphToGraphDef = try LoadFunction(lib, "TF_GraphToGraphDef")
    GraphNextOperation = try LoadFunction(lib, "TF_GraphNextOperation")
    GraphOperationByName = try LoadFunction(lib, "TF_GraphOperationByName")
    OperationGetAttrValueProto = try LoadFunction(lib, "TF_OperationGetAttrValueProto")
    OperationGetAttrTensorList = try LoadFunction(lib, "TF_OperationGetAttrTensorList")
    OperationGetAttrTensor = try LoadFunction(lib, "TF_OperationGetAttrTensor")
    OperationGetAttrTensorShapeProtoList = try LoadFunction(lib, "TF_OperationGetAttrTensorShapeProtoList")
    OperationGetAttrTensorShapeProto = try LoadFunction(lib, "TF_OperationGetAttrTensorShapeProto")
    OperationGetAttrShapeList = try LoadFunction(lib, "TF_OperationGetAttrShapeList")
    OperationGetAttrShape = try LoadFunction(lib, "TF_OperationGetAttrShape")
    OperationGetAttrTypeList = try LoadFunction(lib, "TF_OperationGetAttrTypeList")
    OperationGetAttrType = try LoadFunction(lib, "TF_OperationGetAttrType")
    OperationGetAttrBoolList = try LoadFunction(lib, "TF_OperationGetAttrBoolList")
    OperationGetAttrBool = try LoadFunction(lib, "TF_OperationGetAttrBool")
    OperationGetAttrFloatList = try LoadFunction(lib, "TF_OperationGetAttrFloatList")
    OperationGetAttrFloat = try LoadFunction(lib, "TF_OperationGetAttrFloat")
    OperationGetAttrInt = try LoadFunction(lib, "TF_OperationGetAttrInt")
    OperationGetAttrStringList = try LoadFunction(lib, "TF_OperationGetAttrStringList")
    OperationGetAttrString = try LoadFunction(lib, "TF_OperationGetAttrString")
    OperationGetAttrMetadata = try LoadFunction(lib, "TF_OperationGetAttrMetadata")
    OperationGetControlOutputs = try LoadFunction(lib, "TF_OperationGetControlOutputs")
    OperationNumControlOutputs = try LoadFunction(lib, "TF_OperationNumControlOutputs")
    OperationGetControlInputs = try LoadFunction(lib, "TF_OperationGetControlInputs")
    OperationNumControlInputs = try LoadFunction(lib, "TF_OperationNumControlInputs")
    OperationOutputConsumers = try LoadFunction(lib, "TF_OperationOutputConsumers")
    OperationOutputNumConsumers = try LoadFunction(lib, "TF_OperationOutputNumConsumers")
    OperationInput = try LoadFunction(lib, "TF_OperationInput")
    OperationInputListLength = try LoadFunction(lib, "TF_OperationInputListLength")
    OperationInputType = try LoadFunction(lib, "TF_OperationInputType")
    OperationNumInputs = try LoadFunction(lib, "TF_OperationNumInputs")
    OperationOutputListLength = try LoadFunction(lib, "TF_OperationOutputListLength")
    OperationOutputType = try LoadFunction(lib, "TF_OperationOutputType")
    OperationNumOutputs = try LoadFunction(lib, "TF_OperationNumOutputs")
    OperationDevice = try LoadFunction(lib, "TF_OperationDevice")
    OperationOpType = try LoadFunction(lib, "TF_OperationOpType")
    OperationName = try LoadFunction(lib, "TF_OperationName")
    FinishOperation = try LoadFunction(lib, "TF_FinishOperation")
    SetAttrTensorList = try LoadFunction(lib, "TF_SetAttrTensorList")
    SetAttrTensor = try LoadFunction(lib, "TF_SetAttrTensor")
    SetAttrTensorShapeProtoList = try LoadFunction(lib, "TF_SetAttrTensorShapeProtoList")
    SetAttrTensorShapeProto = try LoadFunction(lib, "TF_SetAttrTensorShapeProto")
    SetAttrShapeList = try LoadFunction(lib, "TF_SetAttrShapeList")
    SetAttrShape = try LoadFunction(lib, "TF_SetAttrShape")
    SetAttrTypeList = try LoadFunction(lib, "TF_SetAttrTypeList")
    SetAttrType = try LoadFunction(lib, "TF_SetAttrType")
    SetAttrBoolList = try LoadFunction(lib, "TF_SetAttrBoolList")
    SetAttrBool = try LoadFunction(lib, "TF_SetAttrBool")
    SetAttrFloatList = try LoadFunction(lib, "TF_SetAttrFloatList")
    SetAttrFloat = try LoadFunction(lib, "TF_SetAttrFloat")
    SetAttrIntList = try LoadFunction(lib, "TF_SetAttrIntList")
    SetAttrInt = try LoadFunction(lib, "TF_SetAttrInt")
    SetAttrStringList = try LoadFunction(lib, "TF_SetAttrStringList")
    SetAttrString = try LoadFunction(lib, "TF_SetAttrString")
    ColocateWith = try LoadFunction(lib, "TF_ColocateWith")
    AddControlInput = try LoadFunction(lib, "TF_AddControlInput")
    AddInputList = try LoadFunction(lib, "TF_AddInputList")
    AddInput = try LoadFunction(lib, "TF_AddInput")
    SetDevice = try LoadFunction(lib, "TF_SetDevice")
    NewOperation = try LoadFunction(lib, "TF_NewOperation")
    GraphGetTensorShape = try LoadFunction(lib, "TF_GraphGetTensorShape")
    GraphGetTensorNumDims = try LoadFunction(lib, "TF_GraphGetTensorNumDims")
    GraphSetTensorShape = try LoadFunction(lib, "TF_GraphSetTensorShape")
    DeleteGraph = try LoadFunction(lib, "TF_DeleteGraph")
    NewGraph = try LoadFunction(lib, "TF_NewGraph")
    DeleteSessionOptions = try LoadFunction(lib, "TF_DeleteSessionOptions")
    SetConfig = try LoadFunction(lib, "TF_SetConfig")
    SetTarget = try LoadFunction(lib, "TF_SetTarget")
    NewSessionOptions = try LoadFunction(lib, "TF_NewSessionOptions")
    StringEncodedSize = try LoadFunction(lib, "TF_StringEncodedSize")
    StringDecode = try LoadFunction(lib, "TF_StringDecode")
    StringEncode = try LoadFunction(lib, "TF_StringEncode")
    TensorData = try LoadFunction(lib, "TF_TensorData")
    TensorByteSize = try LoadFunction(lib, "TF_TensorByteSize")
    Dim = try LoadFunction(lib, "TF_Dim")
    NumDims = try LoadFunction(lib, "TF_NumDims")
    TensorType = try LoadFunction(lib, "TF_TensorType")
    DeleteTensor = try LoadFunction(lib, "TF_DeleteTensor")
    AllocateTensor = try LoadFunction(lib, "TF_AllocateTensor")
    NewTensor = try LoadFunction(lib, "TF_NewTensor")
    DeleteBuffer = try LoadFunction(lib, "TF_DeleteBuffer")
    NewBuffer = try LoadFunction(lib, "TF_NewBuffer")
    NewBufferFromString = try LoadFunction(lib, "TF_NewBufferFromString")
    Message = try LoadFunction(lib, "TF_Message")
    GetCode = try LoadFunction(lib, "TF_GetCode")
    SetStatus = try LoadFunction(lib, "TF_SetStatus")
    DeleteStatus = try LoadFunction(lib, "TF_DeleteStatus")
    NewStatus = try LoadFunction(lib, "TF_NewStatus")
    DataTypeSize = try LoadFunction(lib, "TF_DataTypeSize")
    Version = try LoadFunction(lib, "TF_Version")
  }//end open

  /// static library closing
  public static func Close() {
    if let lib = libDLL {
      _ = dlclose(lib)
    }//end if
  }//end func
}//end class
