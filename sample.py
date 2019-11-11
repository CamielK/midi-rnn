#!/usr/bin/env python
import argparse, os, pdb
import random

import pretty_midi
from datetime import datetime

import train
import utils
import numpy as np

def parse_args():
	parser = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--experiment_dir', type=str,
						default='experiments/default',
						help='directory to load saved model from. ' \
							 'If omitted, it will use the most recent directory from ' \
							 'experiments/.')
	parser.add_argument('--save_dir', type=str,
						help='directory to save generated files to. Directory will be ' \
							 'created if it doesn\'t already exist. If not specified, ' \
							 'files will be saved to generated/ inside --experiment_dir.')
	parser.add_argument('--midi_instrument', default='Acoustic Grand Piano',
						help='MIDI instrument name (or number) to use for the ' \
							 'generated files. See https://www.midi.org/specifications/item/' \
							 'gm-level-1-sound-set for a full list of instrument names.')
	parser.add_argument('--num_files', type=int, default=10,
						help='number of midi files to sample.')
	parser.add_argument('--file_length', type=int, default=100,
						help='Length of each file, measured in 16th notes.')
	parser.add_argument('--prime_file', type=str,
						help='prime generated files from midi file. If not specified ' \
							 'random windows from the validation dataset will be used for ' \
							 'for seeding.')
	parser.add_argument('--data_dir', type=str, default='data/midi',
						help='data directory containing .mid files to use for' \
							 'seeding/priming. Required if --prime_file is not specified')
	parser.add_argument('--use_instrument', type=bool, default=False,
						help='Use instrument type in input.')
	parser.add_argument('--ignore_empty', type=bool, default=False,
						help='Ignore empty windows.')
	parser.add_argument('--encode_section', type=bool, default=False,
						help='Encode source track sections.')
	parser.add_argument('--multi_instruments', type=bool, default=False,
						help='Use multiple instruments to generate a single sample from the prime file.')
	return parser.parse_args()


def get_experiment_dir(experiment_dir):
	if experiment_dir == 'experiments/default':
		dirs_ = [os.path.join('experiments', d) for d in os.listdir('experiments') \
				 if os.path.isdir(os.path.join('experiments', d))]
		experiment_dir = max(dirs_, key=os.path.getmtime)

	if not os.path.exists(os.path.join(experiment_dir, 'model.json')):
		utils.log('Error: {} does not exist. ' \
				  'Are you sure that {} is a valid experiment?' \
				  'Exiting.'.format(os.path.join(args.experiment_dir), 'model.json',
									experiment_dir), True)
		exit(1)

	return experiment_dir


def main():
	args = parse_args()
	args.verbose = True

	# prime file validation
	if args.prime_file and not os.path.exists(args.prime_file):
		utils.log('Error: prime file {} does not exist. Exiting.'.format(args.prime_file),
				  True)
		exit(1)
	else:
		if not os.path.isdir(args.data_dir):
			utils.log('Error: data dir {} does not exist. Exiting.'.format(args.prime_file),
					  True)
			exit(1)

	midi_files = [args.prime_file] if args.prime_file else \
		[os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) \
		 if '.mid' in f or '.midi' in f]

	experiment_dir = get_experiment_dir(args.experiment_dir)
	utils.log('Using {} as --experiment_dir'.format(experiment_dir), args.verbose)

	if not args.save_dir:
		args.save_dir = os.path.join(experiment_dir, 'generated')

	if not os.path.isdir(args.save_dir):
		os.makedirs(args.save_dir)
		utils.log('Created directory {}'.format(args.save_dir), args.verbose)

	model, epoch = train.get_model(args, experiment_dir=experiment_dir)
	utils.log('Model loaded from {}'.format(os.path.join(experiment_dir, 'model.json')),
			  args.verbose)

	window_size = model.layers[0].get_input_shape_at(0)[1]
	seed_generator = utils.get_data_generator(midi_files,
											  window_size=window_size,
											  batch_size=32,
											  num_threads=1,
											  use_instrument=args.use_instrument,
											  ignore_empty=args.ignore_empty,
											  encode_section=args.encode_section,
											  max_files_in_ram=10)

	# validate midi instrument name
	try:
		# try and parse the instrument name as an int
		instrument_num = int(args.midi_instrument)
		if not (instrument_num >= 0 and instrument_num <= 127):
			utils.log('Error: {} is not a supported instrument. Number values must be ' \
					  'be 0-127. Exiting'.format(args.midi_instrument), True)
			exit(1)
		args.midi_instrument = pretty_midi.program_to_instrument_name(instrument_num)
	except ValueError as err:
		# if the instrument name is a string
		try:
			# validate that it can be converted to a program number
			_ = pretty_midi.instrument_name_to_program(args.midi_instrument)
		except ValueError as er:
			utils.log('Error: {} is not a valid General MIDI instrument. Exiting.' \
					  .format(args.midi_instrument), True)
			exit(1)

	if args.multi_instruments:

		if not args.prime_file:
			utils.log('Error: You need to specify a prime file when generating a multi instrument track. Exiting.', True)
			exit(1)

		utils.log(f"Sampling from single seed file: {args.prime_file}", args.verbose)

		generated_midi = pretty_midi.PrettyMIDI(initial_tempo=80)

		source_midi = utils.parse_midi(args.prime_file)

		melody_instruments = source_midi.instruments
		# melody_instruments = utils.filter_monophonic(source_midi.instruments, 1.0)

		for instrument in melody_instruments:
			instrument_group = utils.get_family_id_by_instrument(instrument.program)

			# Get source track seed
			X, y = [], []
			windows = utils._encode_sliding_windows(instrument, window_size)
			for w in windows:
				if np.min(w[0][:, 0]) == 1:
					# Window only contains pauses and Y is also a pause.. ignore!
					continue
				X.append(w[0])
			if len(X) <= 5:
				continue
			seed = X[random.randint(0, len(X) - 1)]

			# Generate track for this instrument
			generated = []
			buf = np.copy(seed).tolist()
			while len(generated) < args.file_length:

				# Add instrument class to input
				if args.use_instrument:
					buf_expanded = [[instrument_group] + x for x in buf]

				# Add section encoding to input
				if args.encode_section:
					sections = [0] * 4
					active_section = int((len(generated) / args.file_length) * 4)
					sections[active_section] = 1
					buf_expanded = [sections + x for x in buf]

				# Get prediction
				arr = np.expand_dims(np.asarray(buf_expanded), 0)
				pred = model.predict(arr)

				# prob distribution sampling
				index = np.random.choice(range(0, seed.shape[1]), p=pred[0])
				pred = np.zeros(seed.shape[1])

				pred[index] = 1
				generated.append(pred)
				buf.pop(0)
				buf.append(pred.tolist())

			# Create instrument
			instrument = utils._network_output_to_instrument(generated, instrument.program)

			# Add to target midi
			generated_midi.instruments.append(instrument)

		if len(generated_midi.instruments) == 0:
			raise Exception(f"Found no monophonic instruments in {args.prime_file}")

		# Save midi
		time = datetime.now().strftime("%Y%m%d%H%M%S")
		sample_name = f"{args.save_dir}/sampled_{time}.mid"
		print(f"Writing generated sample to {sample_name}")
		generated_midi.write(sample_name)

	else:
		# generate 10 tracks using random seeds
		utils.log('Loading seed files...', args.verbose)
		X, y = next(seed_generator)
		generated = utils.generate(model, X, window_size,
								   args.file_length, args.num_files, args.midi_instrument)
		for i, midi in enumerate(generated):
			file = os.path.join(args.save_dir, '{}.mid'.format(i + 1))
			midi.write(file.format(i + 1))
			utils.log('wrote midi file to {}'.format(file), True)


if __name__ == '__main__':
	main()
