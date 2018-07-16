import tensorflow as tf

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
#flags.DEFINE_string("data_path", None, "Where the training/test data is stored.")
flags.DEFINE_string("data_path", "data", "Where the training/test data is stored.")
#flags.DEFINE_string("save_path", None, "Model output directory.")
flags.DEFINE_string("save_path", ".", "Model output directory.")
flags.DEFINE_bool("use_fp16", False, "Train using 16-bit floats instead of 32bit floats")
flags.DEFINE_integer("num_gpus", 0, 
                     "If larger than 1, Grappler AutoParallel optimizer "
                     "will create multiple training replicas with each GPU "
                     "running one replica.")
flags.DEFINE_string("rnn_mode", None,
                    "The low level implementation of lstm cell: one of CUDNN, "
                    "BASIC, and BLOCK, representing cudnn_lstm, basic_lstm, "
                    "and lstm_block_cell classes.")
FLAGS = flags.FLAGS