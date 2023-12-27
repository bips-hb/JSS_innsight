
###############################################################################
#                       Captum: Gradient
###############################################################################
def apply_Gradient(model, inputs, func_args = None, num_outputs = int(1), n_cpu = int(1)):

  import torch
  import torch.nn as nn
  import numpy as np
  import time
  from captum.attr import Saliency, InputXGradient
  
  torch.set_num_threads(int(n_cpu))

  if func_args["times_input"]:
    method = InputXGradient
    args = {}
  else:
    method = Saliency
    args = {'abs': False}
    
  start_time = time.time()
  inputs = torch.tensor(inputs, dtype = torch.float)
  method = method(model)
  convert_time = time.time() - start_time

  inp = inputs.clone()
  inp.requires_grad = True
  input_time = time.time()
  result = list()

  for out_class in range(int(num_outputs)):
    attribution = method.attribute(inp, target = out_class, **args)
    result.append(attribution)
  
  end_time = time.time()
  
  summary = {
    "total_time": end_time - start_time,
    "eval_time": end_time - input_time,
    "convert_time": convert_time,
    "result": torch.stack(result, dim = -1).detach().numpy()
  }
  
  return summary

###############################################################################
#                       Captum: IntegratedGradient
###############################################################################
def apply_IntegratedGradient(model, inputs, func_args = None, num_outputs = int(1), n_cpu = int(1)):

  import torch
  import torch.nn as nn
  import numpy as np
  import time
  from captum.attr import IntegratedGradients
  
  torch.set_num_threads(int(n_cpu))
    
  start_time = time.time()
  inputs = torch.tensor(inputs, dtype = torch.float)
  # for the baseline, we need only a single reference value
  baseline = torch.tensor(func_args['x_ref'], dtype = torch.float)[0].unsqueeze(0)
  
  method = IntegratedGradients(model)
  convert_time = time.time() - start_time

  inp = inputs.clone()
  inp.requires_grad = True
  input_time = time.time()
  result = list()

  for out_class in range(int(num_outputs)):
    attribution = method.attribute(inp, target = out_class, n_steps = int(func_args['n']), baselines = baseline, method = "riemann_right")
    result.append(attribution)
  
  end_time = time.time()
  
  summary = {
    "total_time": end_time - start_time,
    "eval_time": end_time - input_time,
    "convert_time": convert_time,
    "result": torch.stack(result, dim = -1).detach().numpy()
  }
  
  return summary

###############################################################################
#                       Captum: DeepLiftSHAP
###############################################################################
def apply_DeepSHAP(model, inputs, func_args = None, num_outputs = int(1), n_cpu = int(1)):

  import torch
  import torch.nn as nn
  import numpy as np
  import time
  from captum.attr import DeepLiftShap
  
  torch.set_num_threads(int(n_cpu))
    
  start_time = time.time()
  inputs = torch.tensor(inputs, dtype = torch.float)
  baselines = torch.tensor(func_args['x_ref'], dtype = torch.float)
  
  method = DeepLiftShap(model)
  convert_time = time.time() - start_time

  inp = inputs.clone()
  inp.requires_grad = True
  input_time = time.time()
  result = list()

  for out_class in range(int(num_outputs)):
    attribution = method.attribute(inp, target = out_class, baselines = baselines)
    result.append(attribution)
  
  end_time = time.time()
  
  summary = {
    "total_time": end_time - start_time,
    "eval_time": end_time - input_time,
    "convert_time": convert_time,
    "result": torch.stack(result, dim = -1).detach().numpy()
  }
  
  return summary

###############################################################################
#                       Captum: ExpectedGradient
###############################################################################
def apply_ExpectedGradient(model, inputs, func_args = None, num_outputs = int(1), n_cpu = int(1)):

  import torch
  import torch.nn as nn
  import numpy as np
  import time
  from captum.attr import GradientShap
  
  torch.set_num_threads(int(n_cpu))
    
  start_time = time.time()
  inputs = torch.tensor(inputs, dtype = torch.float)
  baselines = torch.tensor(func_args['x_ref'], dtype = torch.float)
  
  method = GradientShap(model)
  convert_time = time.time() - start_time
  
  inp = inputs.clone()
  inp.requires_grad = True
  input_time = time.time()
  result = list()

  for out_class in range(int(num_outputs)):
    attribution = method.attribute(inp, target = out_class, baselines = baselines, n_samples = int(func_args['n']))
    result.append(attribution)
  
  end_time = time.time()
  
  summary = {
    "total_time": end_time - start_time,
    "eval_time": end_time - input_time,
    "convert_time": convert_time,
    "result": torch.stack(result, dim = -1).detach().numpy()
  }
  
  return summary


###############################################################################
#                       Captum: SmoothGrad
###############################################################################
def apply_SmoothGrad(model, inputs, func_args = None, num_outputs = int(1), n_cpu = int(1)):

  import torch
  import torch.nn as nn
  import numpy as np
  import time
  from captum.attr import Saliency, NoiseTunnel, InputXGradient
  
  torch.set_num_threads(int(n_cpu))
  
  # calc stdevs
  std = float((np.max(inputs) - np.min(inputs)) * func_args['noise_level'])

  if func_args["times_input"]:
    method = InputXGradient
    args = {'nt_type': 'smoothgrad', 'nt_samples': int(func_args['n']), 'stdevs': std}
  else:
    method = Saliency
    args = {'abs': False, 'nt_type': 'smoothgrad', 'nt_samples': int(func_args['n']), 'stdevs': std}
    
  start_time = time.time()
  inputs = torch.tensor(inputs, dtype = torch.float)
  method = NoiseTunnel(method(model))
  convert_time = time.time() - start_time

  inp = inputs.clone()
  inp.requires_grad = True
  input_time = time.time()
  result = list()

  for out_class in range(int(num_outputs)):
    attribution = method.attribute(inp, target = out_class, **args)
    result.append(attribution)
  
  end_time = time.time()
  
  summary = {
    "total_time": end_time - start_time,
    "eval_time": end_time - input_time,
    "convert_time": convert_time,
    "result": torch.stack(result, dim = -1).detach().numpy()
  }
  
  return summary

###############################################################################
#                       Captum: LRP
###############################################################################
def apply_LRP(model, inputs, func_args = None, num_outputs = int(1), n_cpu = int(1)):
 
  import torch
  import torch.nn as nn
  import numpy as np
  import time
  from captum.attr import LRP
  from captum.attr._utils.lrp_rules import EpsilonRule, Alpha1_Beta0_Rule
  
  torch.set_num_threads(int(n_cpu))
  
  class Pass(EpsilonRule):
    """
    Pass rule for skipping layer manipulation and propagating the
    relevance over a layer.
    """

    def _create_backward_hook_input(self, inputs):
        def _backward_hook_input(grad):
            return
        return _backward_hook_input
      
    def _create_backward_hook_output(self, outputs):
        def _backward_hook_output(grad):
            return

        return _backward_hook_output

  def set_rules(model, rule_name, rule_param = 0):
    if rule_name == "simple":
      for layer in list(model.modules()):
        if type(layer) in [nn.Linear, nn.Conv2d, nn.AvgPool2d]:
          setattr(layer, 'rule', EpsilonRule(epsilon = 1e-6))
        elif type(layer) in [nn.Flatten]:
          setattr(layer, 'rule', Pass())
        elif type(layer) in [nn.BatchNorm2d, nn.MaxPool2d]:
          setattr(layer, 'rule', EpsilonRule(epsilon = 1e-10))
    elif rule_name == "epsilon":
      for layer in list(model.modules()):
        if type(layer) in [nn.Linear, nn.Conv2d, nn.AvgPool2d]:
          setattr(layer, 'rule', EpsilonRule(epsilon = rule_param))
        elif type(layer) in [nn.Flatten]:
          setattr(layer, 'rule', Pass())
        elif type(layer) in [nn.BatchNorm2d, nn.MaxPool2d]:
          setattr(layer, 'rule', EpsilonRule(epsilon = 1e-10))
  
  start_time = time.time()
  inputs = torch.tensor(inputs, dtype = torch.float)
  set_rules(model, func_args['rule_name'], func_args['rule_param'])
  lrp = LRP(model)
  convert_time = time.time() - start_time

  inp = inputs.clone()
  inp.requires_grad = True
  input_time = time.time()
  result = list()

  for out_class in range(int(num_outputs)):
    attribution = lrp.attribute(inp, target = out_class)
    # Captum deletes all rules after one call
    set_rules(model, func_args['rule_name'], func_args['rule_param'])
    result.append(attribution)
    
  end_time = time.time()

  summary = {
    "total_time": end_time - start_time,
    "eval_time": end_time - input_time,
    "convert_time": convert_time,
    "result": torch.stack(result, dim = -1).detach().numpy()
  }
  
  return summary

###############################################################################
#                       Captum: DeepLift
###############################################################################
def apply_DeepLift(model, inputs, func_args = None, num_outputs = int(1), n_cpu = int(1)):
  import torch
  import torch.nn as nn
  import numpy as np
  import time
  from captum.attr import DeepLift  
  
  torch.set_num_threads(int(n_cpu))
  
  start_time = time.time()
  inputs = torch.tensor(inputs, dtype = torch.float)
  # for the baseline, we need only a single reference value
  baseline = torch.tensor(func_args['x_ref'], dtype = torch.float)[0].unsqueeze(0)
  dl = DeepLift(model, eps = 1e-6)
  convert_time = time.time() - start_time

  inp = inputs.clone()
  inp.requires_grad = True
  input_time = time.time()
  result = list()

  for out_class in range(int(num_outputs)):
    attribution = dl.attribute(inp, baselines = baseline, target=out_class)
    result.append(attribution)
    
  end_time = time.time()
  summary = {
    "total_time": end_time - start_time,
    "eval_time": end_time - input_time,
    "convert_time": convert_time,
    "result": torch.stack(result, dim = -1).detach().numpy()
  }

  return summary

