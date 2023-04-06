
###############################################################################
#                           zennit: Gradient
###############################################################################
def apply_Gradient(model, inputs, func_args = None, num_outputs = int(1), n_cpu = int(1)):
  import torch
  import torch.nn as nn
  import numpy as np
  import time
  from zennit.attribution import Gradient
  
  start_time = time.time()
  inputs = torch.tensor(inputs, dtype = torch.float)
  convert_time = time.time() - start_time

  inp = inputs.clone()
  inp.requires_grad = True
  input_time = time.time()
  result = list()

  for out_class in range(int(num_outputs)):
    # Define attribution function for the output, i.e. only one neuron
    def attr_fun(x):
      ou = torch.zeros_like(x)
      ou[:, out_class] = torch.ones_like(x[:, out_class])
      return ou
    
    # Calculate the gradients
    with Gradient(model, attr_output = attr_fun) as attributor:
      output, relevance = attributor(inp)
      
    if func_args['times_input']:
      relevance = relevance * inp.detach()
    result.append(relevance)
  
  end_time = time.time()
    
  summary = {
    "total_time": end_time - start_time,
    "eval_time": end_time - input_time,
    "convert_time": convert_time,
    "result": torch.stack(result, dim = -1).numpy()
  }

  return summary

###############################################################################
#                           zennit: SmoothGrad
###############################################################################

def apply_SmoothGrad(model, inputs, func_args = None, num_outputs = int(1), n_cpu = int(1)):
  import torch
  import torch.nn as nn
  import numpy as np
  import time
  from zennit.attribution import SmoothGrad
  
  start_time = time.time()
  inputs = torch.tensor(inputs, dtype = torch.float)
  convert_time = time.time() - start_time

  inp = inputs.clone()
  inp.requires_grad = True
  input_time = time.time()
  result = list()

  for out_class in range(int(num_outputs)):
    # Define attribution function for the output, i.e. only one neuron
    def attr_fun(x):
      ou = torch.zeros_like(x)
      ou[:, out_class] = torch.ones_like(x[:, out_class])
      return ou
    
    # Calculate the gradients
    with SmoothGrad(model, attr_output = attr_fun, noise_level = func_args['noise_level'],
                    n_iter = int(func_args['n'])) as attributor:
      output, relevance = attributor(inp)
    result.append(relevance)

  end_time = time.time()
    
  summary = {
    "total_time": end_time - start_time,
    "eval_time": end_time - input_time,
    "convert_time": convert_time,
    "result": torch.stack(result, dim = -1).numpy()
  }
  
  return summary

###############################################################################
#                           zennit: LRP
###############################################################################
def apply_LRP(model, inputs, func_args = None, num_outputs = int(1), n_cpu = int(1)):
  import torch
  import torch.nn as nn
  import numpy as np
  import time
  
  from zennit.composites import LayerMapComposite
  from zennit.rules import Epsilon, Pass, AlphaBeta
  from zennit.types import Convolution, Activation, Linear
  from zennit.attribution import Gradient
  
  
  start_time = time.time()
  if func_args['rule_name'] == "simple":
    def rule():
      return Epsilon(epsilon = 1e-6)
  elif func_args['rule_name'] == "epsilon":
    def rule():
      return Epsilon(epsilon = func_args['rule_param'])
  elif func_args['rule_name'] == "alpha_beta":
    def rule():
      return AlphaBeta(alpha = func_args['rule_param'],
                     beta = func_args['rule_param'] - 1)
                     
  layer_map = [
    (nn.ReLU, Pass()),  # ignore activations
    (nn.Tanh, Pass()),  # ignore activations
    (nn.Conv2d, rule()),
    (nn.Linear, rule()),
    (nn.AvgPool2d, Epsilon(epsilon = 1e-10)),
    (nn.MaxPool2d, Epsilon(epsilon = 1e-10)),
    (nn.BatchNorm2d, Epsilon(epsilon = 1e-10))]
    
  composite = LayerMapComposite(layer_map = layer_map)
  inputs = torch.tensor(inputs, dtype = torch.float)
  convert_time = time.time() - start_time

  inp = inputs.clone()
  inp.requires_grad = True
  input_time = time.time()
  result = list()

  for out_class in range(int(num_outputs)):
    # Define attribution function for the output, i.e. only one neuron
    def attr_fun(x):
      ou = torch.zeros_like(x)
      ou[:, out_class] = x[:, out_class]
      return ou
    
    # Calculate the gradients
    with Gradient(model, composite = composite, attr_output = attr_fun) as attributor:
      output, relevance = attributor(inp)
    
    result.append(relevance)

  end_time = time.time()
    
  summary = {
    "total_time": end_time - start_time,
    "eval_time": end_time - input_time,
    "convert_time": convert_time,
    "result": torch.stack(result, dim = -1).numpy()
  }

  return summary
