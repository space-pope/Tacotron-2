"""
Graph freezing and loading functions.

See:
https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-
serve-it-with-a-python-api-d4f3596b3adc
"""

import os, argparse

import tensorflow as tf

def freeze_graph(model_dir, output_node_names):
	"""Extract the subgraph defined by the output nodes and convert
	all its variables into constants

	Args:
		model_dir: the root folder containing the checkpoint state file
		output_node_names: a string, containing all the output node's names,
							comma separated
	"""
	if not tf.gfile.Exists(model_dir):
		raise AssertionError(
			"Model directory %s doesn't exist. Please specify a "
			"directory with a valid model." % model_dir)

	if not output_node_names:
		print("You need to supply the name of a node to --output_node_names.")
		return -1

	# Retrieve the checkpoint path
	checkpoint = tf.train.get_checkpoint_state(model_dir)
	input_checkpoint = checkpoint.model_checkpoint_path

	# Specify the full filename of the frozen graph
	absolute_model_dir = os.path.dirname(input_checkpoint)
	output_graph = os.path.join(absolute_model_dir, "frozen_model.pb")

	# Clear devices to allow TensorFlow to control on which device
	# it will load operations
	clear_devices = True

	# We start a session using a temporary fresh Graph
	with tf.Session(graph=tf.Graph()) as sess:
		# We import the meta graph in the current default Graph
		saver = tf.train.import_meta_graph(input_checkpoint + '.meta',
										   clear_devices=clear_devices)

		# We restore the weights
		saver.restore(sess, input_checkpoint)

		# print("All ops:")
		# for node in tf.get_default_graph().as_graph_def().node:
		# 	if "inputs" in node.name:
		# 		print(str(node.name))

		# We use a built-in TF helper to export variables to constants
		output_graph_def = tf.graph_util.convert_variables_to_constants(
			# The session is used to retrieve the weights
			sess,
			# The graph_def is used to retrieve the nodes
			tf.get_default_graph().as_graph_def(),
			# The output node names are used to select the useful nodes
			output_node_names.split(",")
		)

		# Finally we serialize and dump the output graph to the filesystem
		with tf.gfile.GFile(output_graph, "wb") as f:
			f.write(output_graph_def.SerializeToString())
		print("%d ops in the final graph." % len(output_graph_def.node))

	return output_graph


def load_graph(frozen_graph_filename):
	"""Load a frozen graph into memory."""
	# We load the protobuf file from the disk and parse it to retrieve the
	# unserialized graph_def
	with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())

	# import the graph_def into a new Graph and return it
	with tf.Graph().as_default() as graph:
		# The name var will prefix every op/node in the graph
		# Since we load everything in a new graph, this is not needed
		tf.import_graph_def(graph_def, name="frozen")
	return graph
	# return graph
