//
//  TensorFlowAPI.h
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

#ifndef __TENSOR_FLOW_STRUCT__
#define __TENSOR_FLOW_STRUCT__
#include <stddef.h>
#include <stdint.h>
// --------------------------------------------------------------------------
// TF_Buffer holds a pointer to a block of data and its associated length.
// Typically, the data consists of a serialized protocol buffer, but other data
// may also be held in a buffer.
//
// By default, TF_Buffer itself does not do any memory management of the
// pointed-to block.  If need be, users of this struct should specify how to
// deallocate the block by setting the `data_deallocator` function pointer.
typedef struct TF_Buffer {
  const void* data;
  size_t length;
  void (*data_deallocator)(void* data, size_t length);
} TF_Buffer;

// --------------------------------------------------------------------------
// The new graph construction API, still under development.

// Represents a computation graph.  Graphs may be shared between sessions.
// Graphs are thread-safe when used as directed below.
typedef struct TF_Graph TF_Graph;

// Operation being built. The underlying graph must outlive this.
typedef struct TF_OperationDescription TF_OperationDescription;

// Operation that has been added to the graph. Valid until the graph is
// deleted -- in particular adding a new operation to the graph does not
// invalidate old TF_Operation* pointers.
typedef struct TF_Operation TF_Operation;

// Represents a specific input of an operation.
typedef struct TF_Input {
  TF_Operation* oper;
  int index;  // The index of the input within oper.
} TF_Input;

// Represents a specific output of an operation.
typedef struct TF_Output {
  TF_Operation* oper;
  int index;  // The index of the output within oper.
} TF_Output;


// TF_AttrMetadata describes the value of an attribute on an operation.
typedef struct TF_AttrMetadata {
  // A boolean: 1 if the attribute value is a list, 0 otherwise.
  unsigned char is_list;

  // Length of the list if is_list is true. Undefined otherwise.
  int64_t list_size;

  // Type of elements of the list if is_list != 0.
  // Type of the single value stored in the attribute if is_list == 0.
  int type;

  // Total size the attribute value.
  // The units of total_size depend on is_list and type.
  // (1) If type == TF_ATTR_STRING and is_list == 0
  //     then total_size is the byte size of the string
  //     valued attribute.
  // (2) If type == TF_ATTR_STRING and is_list == 1
  //     then total_size is the cumulative byte size
  //     of all the strings in the list.
  // (3) If type == TF_ATTR_SHAPE and is_list == 0
  //     then total_size is the number of dimensions
  //     of the shape valued attribute, or -1
  //     if its rank is unknown.
  // (4) If type == TF_ATTR_SHAPE and is_list == 1
  //     then total_size is the cumulative number
  //     of dimensions of all shapes in the list.
  // (5) Otherwise, total_size is undefined.
  int64_t total_size;
} TF_AttrMetadata;

// TF_ImportGraphDefOptions holds options that can be passed to
// TF_GraphImportGraphDef.
typedef struct TF_ImportGraphDefOptions TF_ImportGraphDefOptions;

typedef struct TF_WhileParams {
  // The number of inputs to the while loop, i.e. the number of loop variables.
  // This is the size of cond_inputs, body_inputs, and body_outputs.
  const int ninputs;

  // The while condition graph. The inputs are the current values of the loop
  // variables. The output should be a scalar boolean.
  TF_Graph* const cond_graph;
  const TF_Output* const cond_inputs;
  TF_Output cond_output;

  // The loop body graph. The inputs are the current values of the loop
  // variables. The outputs are the updated values of the loop variables.
  TF_Graph* const body_graph;
  const TF_Output* const body_inputs;
  TF_Output* const body_outputs;

  // Unique null-terminated name for this while loop. This is used as a prefix
  // for created operations.
  const char* name;
} TF_WhileParams;

// --------------------------------------------------------------------------
// API for driving Graph execution.

typedef struct TF_Session TF_Session;
typedef struct TF_DeviceList TF_DeviceList;

// Load plugins containing custom ops and kernels

// TF_Library holds information about dynamically loaded TensorFlow plugins.
typedef struct TF_Library TF_Library;

extern TF_Operation * TF_NextGraphOperation(TF_Graph * graph, size_t * pos);

extern int TF_LoadPatchLibrary(const char * path);
extern void TF_ClosePatchLibrary(void);
#endif
