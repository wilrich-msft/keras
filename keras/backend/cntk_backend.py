#
# This is module emulates a normal backend, but then calls
# into the external CNTK process.
#

import os
import cntk as cn
import numpy as np
from .common import _FLOATX, _EPSILON

import cntk

CNTK_TRAIN_CONFIG_FILENAME = "train.cntk"
CNTK_PREDICT_CONFIG_FILENAME = "predict.cntk"

try:
    import pydot_ng as pydot
    PYDOT = True
except:
    try: 
        import pydot
        PYDOT = True
    except:
        PYDOT = False

_SESSION = None

def _get_session():
    return _SESSION

def _set_session(session):
    global _SESSION
    _SESSION = session

def _name(x):
    if hasattr(x, 'name'):
        name = x.name
    else:
        name = '?'

    if hasattr(x, 'get_shape'):
        name += ' %s'%str(x.get_shape())

    return name

class CNTKTrainConfig(dict):
    # Parsing Keras model and data to come up with necessary
    # template fields for cntk_template.cntk

    def __init__(self, context, keras_model, input_data):
        self.context = context
        self.model = keras_model

        from keras.optimizers import SGD
        if not isinstance(self.model.optimizer, SGD):
            raise ValueError("only SGD is supported on CNTK", self.model.optimizer)
        
        self.X, self.y, self.sample_weight = input_data
        
        # TODO write initialization of InputValue
        # TODO use sample_weight
        self.label_node = cn.Input(self.y.shape, var_name='labels')

        self['ModelPath'] = os.path.join(self.context.directory, 'Model', 'model.dnn')
        self["FeatureDimension"] = self.X.shape[1]
        self["LabelDimension"] = self.y.shape[1]
        last_node_name, model_desc = self._gen_model_description()
        self['ModelDescription'] = model_desc
        self['CriteriaNodes'] = last_node_name
        self['EvalNodes'] = "DUMMY"
        self['OutputNodes'] = "DUMMY"
        self['TrainFile'] = self._get_train_file()
        self['LabelType'] = "regression"
        self['LabelMappingFile'] = self._get_label_mapping_file()        

    def _get_train_file(self):        
        data = np.hstack([self.X, self.y])
        filename = os.path.join(self.context.directory, 'input.txt')
        np.savetxt(filename, data, delimiter=' ', newline='\r\n', fmt='%f')
        return filename

    def _get_label_mapping_file(self):
        data = np.unique(self.y)
        filename = os.path.join(self.context.directory, 'labelMap.txt')
        np.savetxt(filename, data, delimiter=' ', newline='\r\n', fmt='%i')
        return filename

    def _unroll_node(self, output, desc, is_last_layer):
        if is_last_layer:
            if output.name == 'Softmax' and self.model.loss.__name__=='categorical_crossentropy':
                # TODO ultimately this should go into some generalized data structure
                if len(output.params)!=1:
                    raise ValueError("expected exactly one parameter against we would run cross entropy with the labels", output.params)

                output = cn.Operator("CrossEntropyWithSoftmax", 
                        (output.params[0], self.label_node), 
                        get_output_shape=lambda x,y: x.get_shape())


        param_variable_names = []
        if output.params:
            for p in output.params:
                if hasattr(p, 'eval') and p.name:
                    child_var, child_desc = self._unroll_node(p, desc, False)
                    param_variable_names.append(child_var)


        var_name = output.var_name or "v%i"%self.node_counter 
        self.node_counter+=1
        node_name = _name(output)

        params = output.get_cntk_param_string(param_variable_names)

        line = "%s = %s(%s)"%(var_name, output.name, params)
        desc.append((var_name, line))

        return var_name, desc

    def _gen_model_description(self):
        # TODO layer.class_mode should determine the eval node 
        # ('categorical' or 'binary')

        self.node_counter = 0

        _, log = self._unroll_node(self.model.get_output(), [], True)

        # The last node is the criteria node
        criteria_node_name = log[-1][0]

        return criteria_node_name, "\r\n".join(line for var_name, line in log)
    

    def write(self):
        filename = os.path.join(self.context.directory, CNTK_TRAIN_CONFIG_FILENAME)
        tmpl = open(cn.CNTK_TEMPLATE_PATH, "r").read()
        with open(os.path.join(self.context.directory, filename), "w") as out:
            cntk_config_content = tmpl%cntk_config
            out.write(cntk_config_content)
            print("Wrote to directory %s"%self.context.directory)
            
        import subprocess
        subprocess.check_call([cn.CNTK_EXECUTABLE_PATH, "configFile=%s"%filename])

class CNTKPredictConfig(dict):
    
    def __init__(self, context, keras_model, input_data):
        self.context = context
        self.model = keras_model
        
        self.X, self.y = input_data       
        self.y = np.expand_dims(self.y, 1)
        
        self['ModelPath'] = os.path.join(self.context.directory, 'Model', 'model.dnn')
        self["FeatureDimension"] = self.X.shape[1]
        self["LabelDimension"] = self.y.shape[1]
        self['PredictInputFile'] = self._get_test_file()
        self['PredictOutputFile'] = self._get_output_file()        
        self['LabelType'] = "regression"
        self['LabelMappingFile'] = self._get_label_mapping_file()            
    
    def _get_test_file(self):                        
        data = np.hstack([self.X, self.y])
        filename = os.path.join(self.context.directory, 'test.txt')
        np.savetxt(filename, data, delimiter=' ', newline='\r\n', fmt='%f')
        return filename    
        
    def _get_output_file(self):
        return os.path.join(self.context.directory, 'out.txt')
    
    def _get_label_mapping_file(self):        
        return os.path.join(self.context.directory, 'labelMap.txt')

    def write(self):
        filename = os.path.join(self.context.directory, CNTK_PREDICT_CONFIG_FILENAME)
        tmpl = open(cn.CNTK_PREDICT_TEMPLATE_PATH, "r").read()
        with open(os.path.join(self.context.directory, filename), "w") as out:
            cntk_config_content = tmpl%cntk_config
            out.write(cntk_config_content)
            print("Wrote to directory %s"%self.context.directory)
            
        import subprocess
        subprocess.check_call([cn.CNTK_EXECUTABLE_PATH, "configFile=%s"%filename])
        

def write_pydot(g, output, node_counter=0):
    var_name = "v%i"%node_counter 
    node_counter+=1

    param_nodes = []
    if output.params:
        for p in output.params:
            if hasattr(p, 'eval') and p.name:
                param_nodes.append(write_pydot(g, p))

    node_name = _name(output)
    node = pydot.Node(node_name)
    g.add_node(node)
    for var_child, child in param_nodes:
        g.add_edge(pydot.Edge(child, node))

    return var_name, node

def fake_fit(model, ins):
    global cntk_config
    
    if PYDOT:
        g=pydot.Dot()
        g.write_raw("x.dot")
        write_pydot(g, model.get_output())

    with cn.Context(model) as cm:
        cntk_config = CNTKTrainConfig(cm, model, ins)
        cntk_config.write()

def fake_predict(model, ins, verbose=0):
    global cntk_config
    
    with cn.Context(model) as cm:
        cntk_config = CNTKPredictConfig(cm, model, ins)
        cntk_config.write()
    return  [0]
    
def variable(value, dtype=_FLOATX, name=None):
    v = cn.variable(np.asarray(value, dtype=dtype), name=name)
    # TODO initialize
    return v


def placeholder(shape=None, ndim=None, dtype=_FLOATX, name=None):
    if not shape:
        if ndim:
            shape = [None for _ in range(ndim)]

    if len(shape)<2:
        # TODO - we are mixing placeholder for input variables and for weights
        shape.append(None)

    assert len(shape)==2, "Shape %s, expcted 2"%str(shape)
    return cn.placeholder(shape)


def shape(x):
    return x.get_shape()


def ndim(x):
    return len(x.get_shape())

def eval(x):
    '''Run a graph.
    '''
    raise NotImplementedError


def zeros(shape, dtype=_FLOATX, name=None):
    return variable(np.zeros(shape), dtype, name)


def ones(shape, dtype=_FLOATX, name=None):
    return variable(np.ones(shape), dtype, name)


def ones_like(x, name=None):
    raise NotImplementedError

def zeros_like(x, name=None):
    raise NotImplementedError

def cast(x, dtype):
    # TODO implement
    #return cn.cast(x, dtype)
    return x


def dot(x, y):
    return cn.times(x, y)


def transpose(x):
    raise NotImplementedError


def gather(reference, indices):
    '''
    # Arguments
        reference: a tensor.
        indices: an int tensor of indices.

    # Returns
        a tensor of same type as `reference`.
    '''
    raise NotImplementedError


def normalize_axis(axis, ndim):
    if type(axis) is tuple:
        axis = list(axis)
    if type(axis) is list:
        for i, a in enumerate(axis):
            if a is not None and a < 0:
                axis[i] = a % ndim
    else:
        if axis is not None and axis < 0:
            axis = axis % ndim
    return axis


def max(x, axis=None, keepdims=False):
    raise NotImplementedError


def min(x, axis=None, keepdims=False):
    raise NotImplementedError


def sum(x, axis=None, keepdims=False):
    '''Sum of the values in a tensor, alongside the specified axis.
    '''
    raise NotImplementedError


def prod(x, axis=None, keepdims=False):
    '''Multiply the values in a tensor, alongside the specified axis.
    '''
    raise NotImplementedError


def std(x, axis=None, keepdims=False):
    raise NotImplementedError

def mean(x, axis=None, keepdims=False):
    # TODO check axes
    return cn.Operator("Mean", (x,),
            #TODO axis
            get_output_shape=lambda a : a.get_shape()[:-1] # TODO
            )


def any(x, axis=None, keepdims=False):
    '''Bitwise reduction (logical OR).

    Return array of uint8 (0s and 1s).
    '''


def argmax(x, axis=-1):
    if axis < 0:
        axis = axis % len(x.get_shape())
    return cn.argmax(x, axis)


def argmin(x, axis=-1):
    raise NotImplementedError


def square(x):
    raise NotImplementedError

def abs(x):
    raise NotImplementedError


def sqrt(x):
    raise NotImplementedError


def exp(x):
    raise NotImplementedError


def log(x):
    raise NotImplementedError


def round(x):
    raise NotImplementedError


def pow(x, a):
    return cn.log(x)


def clip(x, min_value, max_value):
    return cn.log(x)


def equal(x, y):
    return cn.equal(x, y)


def not_equal(x, y):
    return cn.Operator("**NotEqual**", (x, y))


def maximum(x, y):
    return cn.log(x)


def minimum(x, y):
    return cn.log(x)


def get_value(x):
    '''Technically the same as eval() for cn.
    '''
    raise NotImplementedError

class Function(object):

    def __init__(self, inputs, outputs, updates=[]):
        assert type(inputs) in {list, tuple}
        assert type(outputs) in {list, tuple}
        assert type(updates) in {list, tuple}
        self.inputs = list(inputs)
        self.outputs = list(outputs)
        self.updates = updates

    def __call__(self, inputs):
        return []


def function(inputs, outputs, updates=[]):
    return Function(inputs, outputs, updates=updates)


def gradients(loss, variables):
    return []


def softmax(x):
    return cn.softmax(x)

def categorical_crossentropy(output, target):
    return cn.Operator("CrossEntropy", (output, target), 
            get_output_shape=lambda a,b: a.get_shape()[:-1]
            ) 

