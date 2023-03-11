library(reticulate)

# Create conda environment for TensorFlow 1
conda_create(envname = "JSS_innsight_tf_1", python_version = "3.6.15")
conda_install(envname = "JSS_innsight_tf_1",
              packages = c("tensorflow-cpu==1.15", "keras==2.2.4",
                           "h5py==2.10.0", "deeplift==0.6.13"),
              pip = TRUE)

# Create conda environment for TensorFlow 2
conda_create(envname = "JSS_innsight_tf_2", python_version = "3.8.15")
conda_install(envname = "JSS_innsight_tf_2",
              packages = c("tensorflow-cpu==2.10", "keras",
                           "innvestigate==2.0.2"),
              pip = TRUE)

# Create conda environment for Captum and Zennit
conda_create(envname = "JSS_innsight_pytorch", python_version = "3.8.12")
conda_install(envname = "JSS_innsight_pytorch",
              packages = c("torch", "captum==0.6.0", "zennit==0.5.0"),
              pip_options = c("--extra-index-url https://download.pytorch.org/whl/cpu"),
              pip = TRUE)
