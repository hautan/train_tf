import tensorflow as tf
import argparse
import sys
from model_builder import DrawClassification


def main(unused_args):
    print("Main")
    dc = DrawClassification(FLAGS)
    tf.estimator.train_and_evaluate(dc.estimator, dc.train_spec, dc.eval_spec)
    print(FLAGS)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--training_data",
      type=str,
      default=".data",
      help="Path to training data (tf.Example in TFRecord format)")
  parser.add_argument(
      "--eval_data",
      type=str,
      default=".eval",
      help="Path to evaluation data (tf.Example in TFRecord format)")
  parser.add_argument(
      "--classes_file",
      type=str,
      default="/training/train_tf/rnn/drawing_classification/classes/training.tfrecord.classes",
      help="Path to a file with the classes - one class per line")
  parser.add_argument(
      "--num_layers",
      type=int,
      default=3,
      help="Number of recurrent neural network layers.")
  parser.add_argument(
      "--num_nodes",
      type=int,
      default=128,
      help="Number of node per recurrent network layer.")
  parser.add_argument(
      "--num_conv",
      type=str,
      default="[48, 64, 96]",
      help="Number of conv layers along with number of filters per layer.")
  parser.add_argument(
      "--conv_len",
      type=str,
      default="[5, 5, 3]",
      help="Length of the convolution filters.")
  parser.add_argument(
      "--cell_type",
      type=str,
      default="lstm",
      help="Cell type used for rnn layers: cudnn_lstm, lstm or block_lstm.")
  parser.add_argument(
      "--batch_norm",
      type="bool",
      default="False",
      help="Whether to enable batch normalization or not.")
  parser.add_argument(
      "--learning_rate",
      type=float,
      default=0.0001,
      help="Learning rate used for training.")
  parser.add_argument(
      "--gradient_clipping_norm",
      type=float,
      default=9.0,
      help="Gradient clipping norm used during training.")
  parser.add_argument(
      "--dropout",
      type=float,
      default=0.3,
      help="Dropout used for convolutions and bidi lstm layers.")
  parser.add_argument(
      "--steps",
      type=int,
      default=100000,
      help="Number of training steps.")
  parser.add_argument(
      "--batch_size",
      type=int,
      default=8,
      help="Batch size to use for training/evaluation.")
  parser.add_argument(
      "--model_dir",
      type=str,
      default="/tmp/drawing_classification",
      help="Path for storing the model checkpoints.")
  parser.add_argument(
      "--self_test",
      type="bool",
      default="False",
      help="Whether to enable batch normalization or not.")

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
  
#  classes = []
#  with tf.gfile.GFile(FLAGS.classes_file, "r") as f:
#    classes = [x for x in f]
#  num_classes = len(classes)
#  print(classes)
