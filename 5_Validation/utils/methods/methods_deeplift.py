###############################################################################
#                       Deeplift: Gradient
###############################################################################
def apply_Gradient(model_path, inputs, func_args = None, num_outputs = int(1), n_cpu = int(1)):
  # Load required packages
  import tensorflow as tf
  import keras
  from keras import backend as K
  import numpy as np
  import time
  import os
  
  # packages for deeplift
  import deeplift
  from deeplift.util import compile_func
  from deeplift.layers import NonlinearMxtsMode
  from deeplift.conversion import kerasapi_conversion as kc
  
  tf.compat.v1.disable_eager_execution()
  keras.backend.clear_session()
  config = tf.ConfigProto(intra_op_parallelism_threads = int(n_cpu),
                          inter_op_parallelism_threads = int(n_cpu))
  session = tf.Session(config=config)
  tf.keras.backend.set_session(session)
  
  start_time = time.time()
  # Load model
  model = kc.convert_model_from_saved_files(h5_file = model_path, 
                                            nonlinear_mxts_mode = NonlinearMxtsMode.Gradient)
  # Create scoring function
  if func_args['times_input']:
    score_func = model.get_target_contribs_func(find_scores_layer_idx = 0, 
                                                target_layer_idx = -1) # without last activation
  else:
    score_func = model.get_target_multipliers_func(find_scores_layer_idx = 0, 
                                                   target_layer_idx = -1) # without last activation
  
  convert_time = time.time() - start_time
  input_time = time.time()

  result = list()
  for i in range(int(num_outputs)):
    res = np.array(score_func(
                  task_idx = i,
                  input_data_list = [np.array(inputs)],
                  batch_size = 1000,
                  progress_update = None))
    result.append(res)
  end_time = time.time()

  summary = {
    "total_time": end_time - start_time,
    "eval_time": end_time - input_time,
    "convert_time": convert_time,
    "result": np.stack(result, axis = -1)
  }

  return summary


###############################################################################
#                       Deeplift: DeepLift
###############################################################################
def apply_DeepLift(model_path, inputs, func_args = None, num_outputs = int(1), n_cpu = int(1)):
  # Load required packages
  import keras
  from keras import backend as K
  import tensorflow as tf
  import numpy as np
  import time
  import os
  
  import deeplift
  from deeplift.util import compile_func
  from deeplift.layers import NonlinearMxtsMode
  from deeplift.conversion import kerasapi_conversion as kc
  
  tf.compat.v1.disable_eager_execution()
  keras.backend.clear_session()
  config = tf.ConfigProto(intra_op_parallelism_threads = int(n_cpu),
                          inter_op_parallelism_threads = int(n_cpu))
  session = tf.Session(config=config)
  tf.keras.backend.set_session(session)
  start_time = time.time()
      
  if func_args['rule_name'] == "rescale":
    model = kc.convert_model_from_saved_files(h5_file = model_path, nonlinear_mxts_mode = NonlinearMxtsMode.Rescale)
  elif func_args['rule_name'] == "reveal_cancel":
    model = kc.convert_model_from_saved_files(h5_file = model_path, nonlinear_mxts_mode = NonlinearMxtsMode.RevealCancel)
  else:
    raise ValueError('Unknown rule for DeepLift: ' + func_args['rule_name'])

  score_func = model.get_target_contribs_func(find_scores_layer_idx = 0, target_layer_idx = -1)
  
  convert_time = time.time() - start_time
  input_time = time.time()
  
  result = list()
  for i in range(int(num_outputs)):
    result.append(np.array(score_func(
                  task_idx = i,
                  input_data_list=[inputs],
                  input_references_list=[func_args['x_ref']],
                  batch_size=1000,
                  progress_update=None)))
  end_time = time.time()

  summary = {
    "total_time": end_time - start_time,
    "eval_time": end_time - input_time,
    "convert_time": convert_time,
    "result": np.stack(result, axis = -1)
  }
  
  return summary
