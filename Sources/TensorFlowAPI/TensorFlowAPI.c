//
//  TensorFlowAPI.c
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

#include "TensorFlowAPI.h"
#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>

void * libraryHandle = NULL;
typedef void* (*TF_NewTensor_t)
(int, const int64_t* dims, int num_dims,
void* data, size_t len,void (*deallocator)(void* data, size_t len,
void* arg), void* deallocator_arg);

TF_NewTensor_t TF_NewTensorOrg = NULL;

typedef void (*TF_DeleteTensor_t) (void *);
TF_DeleteTensor_t TF_DeleteTensorOrg = NULL;

typedef TF_Operation * (*TF_GraphNextOperation_t)(TF_Graph * graph, size_t * pos);
TF_GraphNextOperation_t TF_GraphNextOperationOrg = NULL;

TF_Operation * TF_NextGraphOperation(TF_Graph * graph, size_t * pos)
{ return TF_GraphNextOperationOrg && graph && pos
    ? (*TF_GraphNextOperationOrg)(graph, pos) : NULL; }

typedef void * (*TF_NewStatus_t) ();
TF_NewStatus_t TF_NewStatusOrg = NULL;

typedef void (*TF_DeleteStatus_t) (void *);
TF_DeleteStatus_t TF_DeleteStatusOrg = NULL;


int TF_LoadPatchLibrary(const char * path)
{
  if (!path) return -1;
  libraryHandle = dlopen(path, RTLD_LAZY);
  if (!libraryHandle) return -2;
  TF_NewTensorOrg = (TF_NewTensor_t) dlsym(libraryHandle, "TF_NewTensor");
  TF_DeleteTensorOrg = (TF_DeleteTensor_t) dlsym(libraryHandle, "TF_DeleteTensor");
  TF_GraphNextOperationOrg = (TF_GraphNextOperation_t) dlsym(libraryHandle, "TF_GraphNextOperation");
  TF_NewStatusOrg = (TF_NewStatus_t) dlsym(libraryHandle, "TF_NewStatus");
  TF_DeleteStatusOrg = (TF_DeleteStatus_t) dlsym(libraryHandle, "TF_DeleteStatus");
  if (TF_NewTensorOrg && TF_DeleteTensorOrg
      && TF_GraphNextOperationOrg && TF_NewStatusOrg && TF_DeleteStatusOrg)
    return 0;
  return -3;
}

void TF_ClosePatchLibrary(void)
{
  if (libraryHandle) dlclose(libraryHandle);
  libraryHandle = NULL;
  TF_NewTensorOrg = NULL;
  TF_DeleteTensorOrg = NULL;
}
