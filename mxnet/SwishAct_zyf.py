#!/usr/bin/env python
'''
Implementation of Swish Activation Function from Google's paper "SEARCHING FOR ACTIVATION FUNCTIONS" (https://arxiv.org/abs/1710.05941 
)

@author Zhao Yafei (zhaoyafei0210@gmail.com)
@date 2018 Nov. 17
'''

# import numpy as np
import mxnet as mx


class SwishAct(mx.operator.CustomOp):
    def __init__(self, beta=1.0):
        self._beta = beta
        self._x_sig = None
        
    def forward(self, is_train, req, in_data, out_data, aux):
        """Implements forward computation.

        is_train : bool, whether forwarding for training or testing.
        req : list of {'null', 'write', 'inplace', 'add'}, how to assign to out_data. 'null' means skip assignment, etc.
        in_data : list of NDArray, input data.
        out_data : list of NDArray, pre-allocated output buffers.
        aux : list of NDArray, mutable auxiliary states. Usually not used.
        """
        x = in_data[0]
        self._x_sig = mx.nd.sigmoid(x * self._beta)
        y = x * self._x_sig
        self.assign(out_data[0], req[0], y)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        """Implements backward computation

        req : list of {'null', 'write', 'inplace', 'add'}, how to assign to in_grad
        out_grad : list of NDArray, gradient w.r.t. output data.
        in_grad : list of NDArray, gradient w.r.t. input data. This is the output buffer.
        """
        x = in_data[0]
        y = out_data[0]
        dy = out_grad[0]
        dx = y * self._beta + self._x_sig*(1.0 - y * self._beta)
        self.assign(in_grad[0], req[0], dx)

@mx.operator.register("SwishAct")  # register with name "SwishAct"
class SwishActProp(mx.operator.CustomOpProp):
    def __init__(self, beta=1.0):
        super(SwishActProp, self).__init__(True)
        # we use constant bias here to illustrate how to pass arguments
        # to operators. All arguments are in string format so you need
        # to convert them back to the type you want.
        self._beta = float(beta)

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        #  this can be omitted if you only have 1 output.
        return ['output']

    def infer_shape(self, in_shapes):
        """Calculate output shapes from input shapes. This can be
        omited if all your inputs and outputs have the same shape.

        in_shapes : list of shape. Shape is described by a tuple of int.
        """
        data_shape = in_shapes[0]
        output_shape = data_shape
        # return 3 lists representing inputs shapes, outputs shapes, and aux data shapes.
        return (data_shape,), (output_shape,), ()

    def create_operator(self, ctx, in_shapes, in_dtypes):
        #  create and return the CustomOp class.
        return SwishAct(self._beta)

if __name__=='__main__':    
    from mxnet import autograd
    
    x = mx.nd.array([0, 1, 2, 3])
    beta = 1.0 
    # attach gradient buffer to x for autograd
    x.attach_grad()
    print('--->input') 
    print('beta=', beta)
    print('x=', x)
    print('x.grad=', x.grad)

    print('--->forwarding') 
    # forward in a record() section to save computation graph for backward
    # see autograd tutorial to learn more.
    with autograd.record():
        y = mx.nd.Custom(x, beta=beta, op_type='SwishAct')

    print('--->after forwarding') 
    print('y=', y)
    print('y.grad=', y.grad)

    print('--->backwarding') 
    y.backward()
    print('--->after backwarding') 
    print('x.grad=', x.grad)
