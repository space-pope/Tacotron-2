import os
import tensorflow as tf

from hparams import hparams, hparams_debug_string
from tacotron.models import create_model
from tacotron.utils import infolog

tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
FLAGS = tf.app.flags.FLAGS

log = infolog.log


def save(log_dir, args):
	checkpoint_dir = os.path.join(log_dir, 'pretrained/')
	export_dir = os.path.join(log_dir, 'saved/')

	log('Using model: {}'.format(args.model))
	log(hparams_debug_string())


	with tf.variable_scope('datafeeder') as scope:
		inputs = tf.placeholder(tf.int32, shape=(None, None),
											   name='inputs')
		input_lengths = tf.placeholder(tf.int32, shape=(None,),
											   name='input_lengths')

	with tf.variable_scope('model') as scope:
		model = create_model(args.model, hparams)
		model.initialize(inputs, input_lengths)


	#Memory allocation on the GPU as needed
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.gpu_options.per_process_gpu_memory_fraction = args.gpu_fraction

	#Train
	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())

		#saved model restoring
		checkpoint_state = tf.train.get_checkpoint_state(checkpoint_dir)
		if (checkpoint_state and checkpoint_state.model_checkpoint_path):
			log('Loading checkpoint {}'.format(checkpoint_state.model_checkpoint_path))
			saver = tf.train.Saver(max_to_keep=5)
			saver.restore(sess, checkpoint_state.model_checkpoint_path)
		else:
			raise Exception('No model to load at {}'.format(checkpoint_dir))

		# export_path = os.path.join(
		# 	tf.compat.as_bytes(export_dir),
		# 	tf.compat.as_bytes(str(FLAGS.model_version)))
		# print('Exporting trained model to', export_path)
		# builder = tf.saved_model.builder.SavedModelBuilder(export_path)
		print('Exporting trained model to', export_dir)
		builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

		# Build the signature_def_map.
		input_t_info = tf.saved_model.utils.build_tensor_info(inputs)
		input_len_t_info = tf.saved_model.utils.build_tensor_info(input_lengths)
		output_t_info = tf.saved_model.utils.build_tensor_info(model.mel_outputs)

		prediction_signature = (
			tf.saved_model.signature_def_utils.build_signature_def(
				inputs={
					'datafeeder/inputs': input_t_info,
					'datafeeder/input_lengths': input_len_t_info
				},
				outputs={model.mel_outputs.name: output_t_info},
				method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

		legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
		builder.add_meta_graph_and_variables(
			sess, [tf.saved_model.tag_constants.SERVING],
			signature_def_map={
				'inputs': prediction_signature,
				tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
				prediction_signature,
			},
			legacy_init_op=legacy_init_op)

		builder.save()

		print('Done exporting!')

def tacotron_save(args):
	hparams.parse(args.hparams)
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.tf_log_level)
	run_name = args.name or args.model
	log_dir = os.path.join(args.base_dir, 'logs-{}'.format(run_name))
	os.makedirs(log_dir, exist_ok=True)
	infolog.init(os.path.join(log_dir, 'Terminal_train_log'), run_name)
	save(log_dir, args)
