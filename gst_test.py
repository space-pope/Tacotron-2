import argparse
import os
import re
import tensorflow as tf
import time
from tqdm import tqdm

from hparams import hparams, hparams_debug_string
from tacotron.synthesizer import Synthesizer


def run_eval(checkpoint_path, output_dir):
	print(hparams_debug_string())
	synth = Synthesizer()
	eval_dir = os.path.join(output_dir, 'gst-eval')
	log_dir = os.path.join(output_dir, 'gst-logs-eval')

	#Create output path if it doesn't exist
	os.makedirs(eval_dir, exist_ok=True)
	os.makedirs(os.path.join(log_dir, 'wavs'), exist_ok=True)
	os.makedirs(os.path.join(log_dir, 'plots'), exist_ok=True)

	with open(os.path.join(eval_dir, 'map.txt'), 'w') as file:
		for i in tqdm(range(hparams.gst_tokens)):
			hparams.set_hparam('inference_token', i)
			synth.load(checkpoint_path, hp=hparams)
			index = i * 10
			for c, text in enumerate(hparams.gst_sentences):
				index = index + c + 1
				mel_filename = synth.synthesize(text, index, eval_dir,
												log_dir, None)
				file.write('{}|{}\n'.format(text, mel_filename))
			synth.close()

	print('synthesized mel spectrograms at {}'.format(eval_dir))


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--checkpoint', default='logs-Tacotron/pretrained/',
						help='Path to model checkpoint')
	parser.add_argument('--hparams', default='',
						help='Hyperparameter overrides as a comma-separated '+
						' list of name=value pairs')
	parser.add_argument('--output_dir', default='output/',
						help='folder to contain synthesized mel spectrograms')

	args = parser.parse_args()
	hparams.parse(args.hparams)

	try:
		checkpoint_path = tf.train.get_checkpoint_state(
			args.checkpoint).model_checkpoint_path
		print('loaded model at {}'.format(checkpoint_path))
	except:
		raise AssertionError(
			'Cannot restore checkpoint: {}, did you train a model?'.format(
				args.checkpoint))

	run_eval(checkpoint_path, args.output_dir)


if __name__ == '__main__':
	main()
