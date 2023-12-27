
###############################################################################
#                       innvestigate: Gradient
###############################################################################

def apply_Gradient(model_path, inputs, func_args = None, num_outputs = int(1), n_cpu = int(1)):

  # Load required packages
  import tensorflow as tf
  import keras as k
  import innvestigate
  import innvestigate.utils as iutils
  import numpy as np
  import time
  
  tf.compat.v1.disable_eager_execution()
  k.backend.clear_session()
  config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads = int(n_cpu),
                                     inter_op_parallelism_threads = int(n_cpu))
  session = tf.compat.v1.Session(config=config)
  tf.compat.v1.keras.backend.set_session(session)

  # Load model
  model = k.models.load_model(model_path, compile = False)
  
  # Get method (Gradient or GradxInput)
  if func_args['times_input']:
    method_name = "input_t_gradient"
  else:
    method_name = "gradient"

  start_time = time.time()
  analyzer = innvestigate.create_analyzer(method_name, model, neuron_selection_mode = 'index')
  convert_time = time.time() - start_time

  input_time = time.time()
  result = list()
  for i in range(int(num_outputs)):
    result.append(analyzer.analyze(inputs, i))
  end_time = time.time()

  summary = {
    "total_time": end_time - start_time,
    "eval_time": end_time - input_time,
    "convert_time": convert_time,
    "result": np.stack(result, axis = -1)
  }

  return summary

###############################################################################
#                       innvestigate: IntegratedGradient
###############################################################################

def apply_IntegratedGradient(model_path, inputs, func_args = None, num_outputs = int(1), n_cpu = int(1)):

  # Load required packages
  import tensorflow as tf
  import keras as k
  import innvestigate
  import innvestigate.utils as iutils
  import numpy as np
  import time
  
  tf.compat.v1.disable_eager_execution()
  k.backend.clear_session()
  config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads = int(n_cpu),
                                     inter_op_parallelism_threads = int(n_cpu))
  session = tf.compat.v1.Session(config=config)
  tf.compat.v1.keras.backend.set_session(session)

  # Load model
  model = k.models.load_model(model_path, compile = False)
  
  # Get baseline value
  baseline = func_args['x_ref'][[0]]
  
  start_time = time.time()
  analyzer = innvestigate.create_analyzer("integrated_gradients", model, neuron_selection_mode = 'index', steps = int(func_args['n']), reference_inputs = baseline)
  convert_time = time.time() - start_time

  input_time = time.time()
  result = list()
  for i in range(int(num_outputs)):
    result.append(analyzer.analyze(inputs, neuron_selection = int(i)))
  result = analyzer.analyze(inputs)
  end_time = time.time()

  summary = {
    "total_time": end_time - start_time,
    "eval_time": end_time - input_time,
    "convert_time": convert_time,
    "result": np.stack(result, axis = -1)
  }

  return summary

###############################################################################
#                       innvestigate: SmoothGrad
###############################################################################
def apply_SmoothGrad(model_path, inputs, func_args = None, num_outputs = int(1), n_cpu = int(1)):
  # Load required packages
  import tensorflow as tf
  import keras as k
  import innvestigate
  import innvestigate.utils as iutils
  import numpy as np
  import time
  
  tf.compat.v1.disable_eager_execution()
  k.backend.clear_session()
  config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads = int(n_cpu),
                                     inter_op_parallelism_threads = int(n_cpu))
  session = tf.compat.v1.Session(config=config)
  tf.compat.v1.keras.backend.set_session(session)

  # Load model
  model = k.models.load_model(model_path, compile = False)

  start_time = time.time()
  analyzer = innvestigate.create_analyzer("smoothgrad", model, 
    augment_by_n = int(func_args['n']), noise_scale = func_args['noise_level'],
    neuron_selection_mode = 'index')
  convert_time = time.time() - start_time

  input_time = time.time()

  result = list()
  for i in range(int(num_outputs)):
    result.append(analyzer.analyze(inputs, neuron_selection = i))
  end_time = time.time()

  summary = {
    "total_time": end_time - start_time,
    "eval_time": end_time - input_time,
    "convert_time": convert_time,
    "result": np.stack(result, axis = -1)
  }

  return summary

###############################################################################
#                       innvestigate: LRP
###############################################################################
def apply_LRP(model_path, inputs, func_args = None, num_outputs = int(1), n_cpu = int(1)):
  # Load required packages
  import tensorflow as tf
  import keras as k
  import innvestigate
  import innvestigate.utils as iutils
  import numpy as np
  import time
  
  tf.compat.v1.disable_eager_execution()
  k.backend.clear_session()
  config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads = int(n_cpu),
                                     inter_op_parallelism_threads = int(n_cpu))
  session = tf.compat.v1.Session(config=config)
  tf.compat.v1.keras.backend.set_session(session)

  # Load model
  model = k.models.load_model(model_path, compile = False)

  if func_args['rule_name'] == 'simple':
    method_name = "lrp.epsilon"
    args = {"epsilon": 1e-6}
  elif func_args['rule_name'] == 'epsilon':
    method_name = "lrp.epsilon"
    args = {"epsilon": func_args['rule_param']}
  elif func_args['rule_name'] == 'alpha_beta':
    if func_args['rule_param'] == 1:
      method_name = "lrp.alpha_1_beta_0"
      args = {}
    else:
      method_name = "lrp.alpha_beta"
      args = {"alpha": func_args['rule_param']}
    
  start_time = time.time()
  analyzer = innvestigate.create_analyzer(method_name, model,
    neuron_selection_mode = 'index', **args)
  convert_time = time.time() - start_time
  
  input_time = time.time()
  result = list()
  for i in range(int(num_outputs)):
    result.append(analyzer.analyze(inputs, i))
  end_time = time.time()

  summary = {
    "total_time": end_time - start_time,
    "eval_time": end_time - input_time,
    "convert_time": convert_time,
    "result": np.stack(result, axis = -1)
  }

  return summary
