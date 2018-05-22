import os
import argparse
from tacotron.save_model import tacotron_save


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--base_dir', default=os.path.expanduser('~/data'))
	parser.add_argument('--hparams', default='',
		help='Hyperparameter overrides as a comma-separated list of name=value pairs')
	parser.add_argument('--name', help='Name of logging directory.')
	parser.add_argument('--model', default='Tacotron', choices=['Tacotron'])
	parser.add_argument('--tf_log_level', type=int, default=1, help='Tensorflow C++ log level.')
	parser.add_argument('--gpu_fraction', type=float, default=0.8, help='Fraction of GPU memory to allocate')
	args = parser.parse_args()
	tacotron_save(args)


if __name__ == '__main__':
	main()
