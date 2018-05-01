"""
Simple test script to measure memory usage of a frozen model.

Observe GPU utilization using nvidia-smi while the script runs.

Adapted from
https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-
serve-it-with-a-python-api-d4f3596b3adc
"""

import argparse
import numpy as np
import tensorflow as tf
from hparams import hparams
from tacotron.utils.graph_io import freeze_graph, load_graph
from tacotron.utils.text import text_to_sequence

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--input",
						default=None,
						type=str,
						help="Directory containing checkpoints from which to \
						freeze then load a model")
	parser.add_argument("--frozen_model",
						default=None,
						type=str,
						help="Frozen model file to import")
	args = parser.parse_args()

	model_file = args.frozen_model
	output_nodes = "datafeeder/inputs,datafeeder/input_lengths,"\
				   "model/inference/add"

	if args.input:
		model_file = freeze_graph(args.input, output_nodes)
		print("Graph frozen to {}".format(model_file))

	graph = load_graph(model_file)
	print("Graph loaded from {}".format(model_file))

	cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
	text = "This is a text sample for running inference. It's" \
	" not too long; it's not too short; it's just right."
	seq = text_to_sequence(text, cleaner_names)

	inputs = graph.get_tensor_by_name('frozen/datafeeder/inputs:0')
	input_lengths = graph.get_tensor_by_name('frozen/datafeeder/input_lengths:0')
	feed_dict = {
		inputs: [np.asarray(seq, dtype=np.int32)],
		input_lengths: np.asarray([len(seq)], dtype=np.int32),
	}
	# We access the input and output nodes
	y = graph.get_tensor_by_name('frozen/model/inference/add:0')

	# Launch a Session
	with tf.Session(graph=graph) as sess:
		# Note: we don't need to initialize/restore anything
		# There are no Variables in this graph, only hardcoded constants
		y_out = sess.run(y, feed_dict=feed_dict)
		print(y_out)
