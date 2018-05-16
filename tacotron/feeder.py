import numpy as np
import os
import random
import threading
import time
import traceback
from tacotron.utils.cmudict import CMUDict
from tacotron.utils.text import text_to_sequence
from tacotron.utils.infolog import log
import tensorflow as tf
from hparams import hparams


_batches_per_group = 32

# probability that an IPA pronunciation will be substituted for a word in the
# training data
_p_cmudict = 0.5

# pad input sequences with the <pad_token> 0 ( _ )
_pad = 0

# explicitly setting the padding to a value that doesn't originally exist in
# the spectogram to avoid any possible conflicts, without affecting the output
# range of the model too much
if hparams.symmetric_mels:
	_target_pad = -(hparams.max_abs_value + .1)
else:
	_target_pad = -0.1

# Mark finished sequences with 1s
_token_pad = 1.

class Feeder(threading.Thread):
	"""
		Feeds batches of data into queue on a background thread.
	"""

	def __init__(self, coordinator, metadata_filename, hparams):
		super(Feeder, self).__init__()
		self._coord = coordinator
		self._hparams = hparams
		self._cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
		self._offset = 0
		self._cmudict = hparams.tacotron_use_cmudict

		# Load metadata
		data_dir = os.path.dirname(metadata_filename)
		self._mel_dir = os.path.join(data_dir, 'mels')
		self._linear_dir = os.path.join(data_dir, 'linear')
		self._test_metadata = []
		with open(metadata_filename, encoding='utf-8') as f:
			self._metadata = [line.strip().split('|') for line in f]
			frame_shift_ms = hparams.hop_size / hparams.sample_rate
			hours = sum([int(x[4]) for x in self._metadata]) \
					* frame_shift_ms / (3600)
			log('Loaded metadata for {} examples ({:.2f} hours)'.format(
				len(self._metadata), hours))

		if hparams.tacotron_test_batches > 0:
			self._metadata, self._test_metadata = _split_data(self._metadata,
															  hparams)

		# Create placeholders for inputs and targets. Don't specify batch size
		# because we want to be able to feed different batch sizes at eval time.
		self._placeholders = [
			tf.placeholder(tf.int32, shape=(None, None), name='inputs'),
			tf.placeholder(tf.int32, shape=(None, ), name='input_lengths'),
			tf.placeholder(tf.float32, shape=(None, None, hparams.num_mels),
						   name='mel_targets'),
			tf.placeholder(tf.float32, shape=(None, None),
						   name='token_targets'),
			tf.placeholder(tf.float32, shape=(None, None, hparams.num_freq),
						   name='linear_targets'),
		]

		# Create queue for buffering data
		queue = tf.FIFOQueue(8, [tf.int32, tf.int32, tf.float32, tf.float32,
								 tf.float32], name='input_queue')
		self._enqueue_op = queue.enqueue(self._placeholders)
		(self.inputs, self.input_lengths, self.mel_targets, self.token_targets,
		 self.linear_targets) = queue.dequeue()
		self.inputs.set_shape(self._placeholders[0].shape)
		self.input_lengths.set_shape(self._placeholders[1].shape)
		self.mel_targets.set_shape(self._placeholders[2].shape)
		self.token_targets.set_shape(self._placeholders[3].shape)
		self.linear_targets.set_shape(self._placeholders[4].shape)

		# Load CMUDict: If enabled, this will randomly substitute some words in
		# the training data with their IPA equivalents, which will allow you to
		# also pass IPA to the model for synthesis
		# (useful for proper nouns, etc.)
		if hparams.tacotron_use_cmudict:
			cmudict_path = os.path.join(data_dir, 'cmudict-0.7b-ipa.txt')
			if not os.path.isfile(cmudict_path):
				cmu_host = 'https://raw.githubusercontent.com/menelik3/'+ \
						   'cmudict-ipa/master/cmudict-0.7b-ipa.txt'
				raise Exception(
					'If tacotron_use_cmudict=True, you must download ' +
					'{} to {}'.format(cmu_host, cmu_path))
			self._cmudict = CMUDict(cmudict_path, keep_ambiguous=False)
			log('Loaded CMUDict with {} unambiguous entries'.format(
				len(self._cmudict)))
		else:
			self._cmudict = None

	def limit_data(self, max_hours, hparams):
		cutoff = len(self._metadata)
		total = 0
		frame_shift_ms = hparams.hop_size / hparams.sample_rate
		adjusted_max = max_hours * 3600 / frame_shift_ms
		for i, x in enumerate(self._metadata):
			total += int(x[4])
			if total >= adjusted_max:
				cutoff = i
				break
		self._metadata = self._metadata[:cutoff + 1]
		hours = total * frame_shift_ms / 3600
		log('Limited metadata to %d examples (%.2f hours)' % (cutoff, hours))

	def start_in_session(self, session):
		self._session = session
		self.start()

	def run(self):
		try:
			while not self._coord.should_stop():
				self._enqueue_next_group()
		except Exception as e:
			traceback.print_exc()
			self._coord.request_stop(e)

	def test_data(self):
		"""
		Generates a list of test batches.
		"""
		# Read a group of examples:
		n = self._hparams.tacotron_batch_size
		r = self._hparams.outputs_per_step
		examples = [self._load_example(i, self._test_metadata)
					for i in range(0, len(self._test_metadata))]
		batches = _batch_examples(examples, n)

		for batch in batches:
			yield dict(zip(self._placeholders,
						   _prepare_batch(batch, r)[:-1]))

	def _enqueue_next_group(self):
		start = time.time()

		# Read a group of examples
		n = self._hparams.tacotron_batch_size
		r = self._hparams.outputs_per_step
		examples = [self._get_next_example()
					for i in range(n * _batches_per_group)]

		batches = _batch_examples(examples, n)
		log('\nGenerated {} batches of size {} in {:.3f} sec'.format(
			len(batches), n, time.time() - start))
		for batch in batches:
			feed_dict = dict(zip(self._placeholders, _prepare_batch(batch, r)))
			self._session.run(self._enqueue_op, feed_dict=feed_dict)

	def _get_next_example(self):
		"""
		Gets a single example (input, mel_target, token_target, linear_target,
		mel_length) from disk
		"""
		if self._offset >= len(self._metadata):
			self._offset = 0
			np.random.shuffle(self._metadata)
		example = self._load_example(self._offset, self._metadata)
		self._offset += 1
		return example

	def _load_example(self, index, metadata):
		meta = metadata[index]
		text = meta[5]
		if self._cmudict and random.random() < _p_cmudict:
			text = ' '.join([self._maybe_get_ipa(word)
							 for word in text.split(' ')])

		input_data = np.asarray(text_to_sequence(text, self._cleaner_names),
								dtype=np.int32)
		mel_target = np.load(os.path.join(self._mel_dir, meta[1]))
		# Create parallel sequences containing zeros to represent a non
		# finished sequence
		token_target = np.asarray([0.] * len(mel_target))
		linear_target = np.load(os.path.join(self._linear_dir, meta[2]))
		return (input_data, mel_target, token_target, linear_target,
				len(mel_target))

	def _maybe_get_ipa(self, word):
		strip_emphasis = random.random() < 0.7
		ipa = self._cmudict.lookup(word, strip_emphasis)
		return '{%s}' % ipa[0] \
			if ipa is not None and random.random() < 0.5 else word


def _batch_examples(examples, n):
  # Bucket examples based on similar output sequence length for efficiency:
  examples.sort(key=lambda x: x[-1])
  batches = [examples[i:i+n] for i in range(0, len(examples), n)]
  np.random.shuffle(batches)
  return batches


def _prepare_batch(batch, outputs_per_step):
	np.random.shuffle(batch)
	inputs = _prepare_inputs([x[0] for x in batch])
	input_lengths = np.asarray([len(x[0]) for x in batch], dtype=np.int32)
	mel_targets = _prepare_targets([x[1] for x in batch], outputs_per_step)
	# Pad sequences with 1 to infer that the sequence is done
	token_targets = _prepare_token_targets([x[2] for x in batch],
										   outputs_per_step)
	linear_targets = _prepare_targets([x[3] for x in batch], outputs_per_step)
	return (inputs, input_lengths, mel_targets, token_targets, linear_targets)

def _prepare_inputs(inputs):
	max_len = max([len(x) for x in inputs])
	return np.stack([_pad_input(x, max_len) for x in inputs])

def _prepare_targets(targets, alignment):
	max_len = max([len(t) for t in targets]) + 1
	return np.stack([_pad_target(t, _round_up(max_len, alignment))
					 for t in targets])

def _prepare_token_targets(targets, alignment):
	max_len = max([len(t) for t in targets]) + 1
	return np.stack([_pad_token_target(t, _round_up(max_len, alignment))
					 for t in targets])

def _pad_input(x, length):
	return np.pad(x, (0, length - x.shape[0]), mode='constant',
				  constant_values=_pad)

def _pad_target(t, length):
	return np.pad(t, [(0, length - t.shape[0]), (0, 0)], mode='constant',
				  constant_values=_target_pad)

def _pad_token_target(t, length):
	return np.pad(t, (0, length - t.shape[0]), mode='constant',
				  constant_values=_token_pad)

def _round_up(x, multiple):
	remainder = x % multiple
	return x if remainder == 0 else x + multiple - remainder

def _split_data(metadata, hparams):
	""" Splits a data set into a tuple of train and test samples. """
	test_batches = hparams.tacotron_test_batches
	batch_size = hparams.tacotron_batch_size
	total_samples = len(metadata)
	while batch_size * test_batches > total_samples:
		test_batches -= 1
	test_samples = test_batches * batch_size
	return metadata[:-test_samples], metadata[-test_samples:]
