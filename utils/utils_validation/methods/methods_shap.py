
###############################################################################
#                       SHAP: DeepLiftSHAP
###############################################################################
def apply_DeepSHAP(model, inputs, func_args = None, num_outputs = int(1), n_cpu = int(1)):

  import torch
  import torch.nn as nn
  import numpy as np
  import time
  import shap
  
  torch.set_num_threads(int(n_cpu))
    
  start_time = time.time()
  baselines = torch.tensor(func_args['x_ref'], dtype = torch.float)
  inputs = torch.tensor(inputs, dtype = torch.float)
  
  method = shap.DeepExplainer(model, baselines)
  convert_time = time.time() - start_time
  
  input_time = time.time()
  result = list()

  shap_values = method.shap_values(inputs)
  
  if type(shap_values) != list:
    shap_values = [shap_values]
  
  end_time = time.time()
  
  summary = {
    "total_time": end_time - start_time,
    "eval_time": end_time - input_time,
    "convert_time": convert_time,
    "result": np.stack(shap_values, axis = -1)
  }
  
  return summary


###############################################################################
#                       SHAP: ExpectedGradient
###############################################################################
def apply_ExpectedGradient(model, inputs, func_args = None, num_outputs = int(1), n_cpu = int(1)):

  import torch
  import torch.nn as nn
  import numpy as np
  import time
  import shap
  
  torch.set_num_threads(int(n_cpu))
    
  start_time = time.time()
  baselines = torch.tensor(func_args['x_ref'], dtype = torch.float)
  inputs = torch.tensor(inputs, dtype = torch.float)
  
  method = shap.GradientExplainer(model, baselines)
  convert_time = time.time() - start_time
  
  input_time = time.time()
  result = list()
  
  shap_values = method.shap_values(inputs, nsamples = int(func_args['n']))
  
  if type(shap_values) != list:
    shap_values = [shap_values]
  
  end_time = time.time()
  
  summary = {
    "total_time": end_time - start_time,
    "eval_time": end_time - input_time,
    "convert_time": convert_time,
    "result": np.stack(shap_values, axis = -1)
  }
  
  return summary
