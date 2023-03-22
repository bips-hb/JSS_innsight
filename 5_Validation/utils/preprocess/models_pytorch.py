
###############################################################################
#                           PyTorch dense models
###############################################################################

def get_dense_model(shape, name, save = True, act = "relu", bias = True, num_outputs = 1, 
                    src_dir = "models", depth = 1, width = 64):
  import torch
  import torch.nn as nn
  
  torch.set_num_threads(int(1))
  
  depth = int(depth)
  width = int(width)
  
  ## Define model
  # first layer
  model = nn.Sequential(
    nn.Linear(int(shape), width,  bias = bias)
  )
  if act == "relu":
    model.add_module("first_act", nn.ReLU())
  elif act == "tanh":
    model.add_module("first_act", nn.Tanh())
    
  for i in range(depth - 1):
    model.add_module("layer_" + str(i + 1), nn.Linear(width, width, bias = bias))
    if act == "relu":
      model.add_module("act_" + str(i + 1), nn.ReLU())
    elif act == "tanh":
      model.add_module("act_" + str(i + 1), nn.Tanh())
    
  # second layer
  model.add_module("last_layer", nn.Linear(width, int(num_outputs), bias = bias))
  
  # save model
  if save:
    torch.save(dict(model.state_dict()), src_dir + "/models/" + name, 
                _use_new_zipfile_serialization = True)
  
  return model

###############################################################################
#                           PyTorch CNN models
###############################################################################

def get_2D_model(shape, name, save = True, act = "relu", bias = True, 
    pooling = "none", bn = "none", num_outputs = 5, src_dir = "models",
    depth = 1, width = 5):
  import torch
  import torch.nn as nn
  
  torch.set_num_threads(int(1))
  
  in_channels = int(shape[0])
  depth = int(depth)
  width = int(width)
  
  if act == "relu":
    activation = nn.ReLU
  elif act == "tanh":
    activation = nn.Tanh
    
  # Define model
  model = nn.Sequential()
  num_channels = in_channels
  
  for i in range(depth):
    if i == depth - 1:
      kernel_size = [4, 4]
      padding = 0
    else:
      kernel_size = [3,3]
      padding = 1
    model.add_module("layer_" + str(i+1), nn.Conv2d(num_channels, width, kernel_size, bias = bias, padding = padding))
    num_channels = width
    
    # add batchnorm layer
    if bn == "none":
      model.add_module("act_" + str(i+1), activation())
    elif bn == "after_act":
      model.add_module("act_" + str(i+1), activation())
      model.add_module("batchnorm_" + str(i+1), nn.BatchNorm2d(width, affine = bias))
    elif bn == "before_act":
      model.add_module("batchnorm_" + str(i+1), nn.BatchNorm2d(width, affine = bias))
      model.add_module("act_" + str(i+1), activation())
  
  
  # add pooling layer
  if pooling == "avg":
    model.add_module("AvgPooling_1", nn.AvgPool2d([3,3]))
  elif pooling == "max":
    model.add_module("MaxPooling_1", nn.MaxPool2d([3,3]))
  
  # add flatten
  model.add_module("flatten", nn.Flatten())
  
  # Calc output of flatten
  inputs = torch.randn([1, in_channels, int(shape[1]), int(shape[2])])
  out = model(inputs)
  
  # add last layer
  model.add_module("output", nn.Linear(out.shape[1], int(num_outputs), bias = bias))
  
  model.eval()
  
  if save:
    torch.save(dict(model.state_dict()), src_dir + "/models/" + name,
                _use_new_zipfile_serialization = True)
  
  return model
