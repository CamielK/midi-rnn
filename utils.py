import os, glob, random
import pretty_midi
import numpy as np
from collections import defaultdict
from keras.models import model_from_json
from multiprocessing import Pool as ThreadPool
import json


def log(message, verbose):
	if verbose:
		print('[*] {}'.format(message))


def parse_midi(path):
	midi = None
	try:
		midi = pretty_midi.PrettyMIDI(path)
		midi.remove_invalid_notes()
	except Exception as e:
		raise Exception(("%s\nerror readying midi file %s" % (e, path)))
	return midi


def get_percent_monophonic(pm_instrument_roll):
	mask = pm_instrument_roll.T > 0
	notes = np.sum(mask, axis=1)
	n = np.count_nonzero(notes)
	single = np.count_nonzero(notes == 1)
	if single > 0:
		return float(single) / float(n)
	elif single == 0 and n > 0:
		return 0.0
	else:  # no notes of any kind
		return 0.0


def filter_monophonic(pm_instruments, percent_monophonic=0.99):
	return [i for i in pm_instruments if \
			get_percent_monophonic(i.get_piano_roll()) >= percent_monophonic]


# if the experiment dir doesn't exist create it and its subfolders
def create_experiment_dir(experiment_dir, verbose=False):
	# if the experiment directory was specified and already exists
	if experiment_dir != 'experiments/default' and \
			os.path.exists(experiment_dir):
		# raise an error
		raise Exception('Error: Invalid --experiment_dir, {} already exists' \
						.format(experiment_dir))

	# if the experiment directory was not specified, create a new numeric folder
	if experiment_dir == 'experiments/default':

		experiments = os.listdir('experiments')
		experiments = [dir_ for dir_ in experiments \
					   if os.path.isdir(os.path.join('experiments', dir_))]

		most_recent_exp = 0
		for dir_ in experiments:
			try:
				most_recent_exp = max(int(dir_), most_recent_exp)
			except ValueError as e:
				# ignrore non-numeric folders in experiments/
				pass

		experiment_dir = os.path.join('experiments',
									  str(most_recent_exp + 1).rjust(2, '0'))

	os.mkdir(experiment_dir)
	log('Created experiment directory {}'.format(experiment_dir), verbose)
	os.mkdir(os.path.join(experiment_dir, 'checkpoints'))
	log('Created checkpoint directory {}'.format(os.path.join(experiment_dir, 'checkpoints')),
		verbose)
	os.mkdir(os.path.join(experiment_dir, 'tensorboard-logs'))
	log('Created log directory {}'.format(os.path.join(experiment_dir, 'tensorboard-logs')),
		verbose)

	return experiment_dir


# load data from prepared datset containing instrument tracks
# Shuffle batches should be false for training!!
def get_prepared_data_generator(all_tracks, window_size=20, batch_size=32,
					   use_instrument=False, ignore_empty=False, encode_section=False,
					   max_tracks_in_ram=170, shuffle_batches=False):
	load_index = 0

	while True:

		if not shuffle_batches:
			tracks = all_tracks[load_index:load_index + max_tracks_in_ram]
			load_index = (load_index + max_tracks_in_ram) % len(tracks)
		else:
			# Select a random subset of tracks, this should only be used for validation
			tracks = random.sample(all_tracks, max_tracks_in_ram)

		# Get windows from tracks
		# print('Finished in {:.2f} seconds'.format(time.time() - start_time))
		# print('parsed, now extracting data')
		data = _windows_from_tracks(tracks, window_size, use_instrument, ignore_empty, encode_section)
		batch_index = 0
		while batch_index + batch_size < len(data[0]):
			# print('getting data...')
			# print('yielding small batch: {}'.format(batch_size))

			res = (data[0][batch_index: batch_index + batch_size],
				   data[1][batch_index: batch_index + batch_size])
			yield res
			batch_index = batch_index + batch_size

		# probably unneeded but why not
		del tracks  # free the mem
		del data  # free the mem


# load data with a lazzy loader
def get_data_generator(midi_paths,
					   window_size=20,
					   batch_size=32,
					   num_threads=8,
					   use_instrument=False,
					   ignore_empty=False,
					   encode_section=False,
					   max_files_in_ram=170):
	if num_threads > 1:
		# load midi data
		pool = ThreadPool(num_threads)

	load_index = 0

	while True:
		load_files = midi_paths[load_index:load_index + max_files_in_ram]
		# print('length of load files: {}'.format(len(load_files)))
		load_index = (load_index + max_files_in_ram) % len(midi_paths)

		# print('loading large batch: {}'.format(max_files_in_ram))
		# print('Parsing midi files...')
		# start_time = time.time()
		if num_threads > 1:
			parsed = pool.map(parse_midi, load_files)
		else:
			parsed = map(parse_midi, load_files)
		# print('Finished in {:.2f} seconds'.format(time.time() - start_time))
		# print('parsed, now extracting data')
		data = _windows_from_monophonic_instruments(parsed, window_size, use_instrument, ignore_empty, encode_section)
		batch_index = 0
		while batch_index + batch_size < len(data[0]):
			# print('getting data...')
			# print('yielding small batch: {}'.format(batch_size))

			res = (data[0][batch_index: batch_index + batch_size],
				   data[1][batch_index: batch_index + batch_size])
			yield res
			batch_index = batch_index + batch_size

		# probably unneeded but why not
		del parsed  # free the mem
		del data  # free the mem


def save_model(model, model_dir):
	with open(os.path.join(model_dir, 'model.json'), 'w') as f:
		f.write(model.to_json())


def load_model_from_checkpoint(model_dir):
	'''Loads the best performing model from checkpoint_dir'''
	with open(os.path.join(model_dir, 'model.json'), 'r') as f:
		model = model_from_json(f.read())

	epoch = 0
	newest_checkpoint = max(glob.iglob(model_dir +
									   '/checkpoints/*.hdf5'),
							key=os.path.getctime)

	if newest_checkpoint:
		epoch = int(newest_checkpoint[len(newest_checkpoint) - 8:len(newest_checkpoint) - 5])
		load_checkpoint(model, newest_checkpoint)

	return model, epoch


def load_checkpoint(model, checkpoint):
	model.load_weights(checkpoint)


def generate(model, seeds, window_size, length, num_to_gen, instrument_name, use_instrument = False, encode_section = False):
	# generate a pretty midi file from a model using a seed
	def _gen(model, seed, window_size, length, use_instrument = False, encode_section = False):

		output_size = seed.shape[1]
		if use_instrument:
			output_size -= 1
		if encode_section:
			output_size -= 4

		generated = []
		# ring buffer
		buf = np.copy(seed).tolist()
		if encode_section:
			instrument = buf[0][4]
		else:
			instrument = buf[0][0]
		while len(generated) < length:
			buf_expanded = [x for x in buf]

			# Add instrument class to input only on first run
			if use_instrument:
				buf_expanded = [[instrument] + x if len(x)==output_size else x for x in buf_expanded]

			# Add section encoding to input
			if encode_section:
				sections = [0] * 4
				active_section = int((len(generated) / length) * 4)
				sections[active_section] = 1
				buf_expanded = [sections + x if len(x)<=output_size+1 else x for x in buf_expanded]

			arr = np.expand_dims(np.asarray(buf_expanded), 0)
			pred = model.predict(arr)

			# argmax sampling (NOT RECOMMENDED), or...
			# index = np.argmax(pred)

			# prob distrobuition sampling
			index = np.random.choice(range(0, output_size), p=pred[0])
			pred = np.zeros(output_size)

			pred[index] = 1
			generated.append(pred)
			buf.pop(0)
			buf.append(pred.tolist())

		instrument_program = None
		if use_instrument:
			instrument_program = get_family_instrument_by_normalized_class(instrument)  # Convert from normalized family class back to instrument
		return generated, instrument_program

	midis = []
	for i in range(0, num_to_gen):
		seed = seeds[random.randint(0, len(seeds) - 1)]
		gen, instrument_program = _gen(model, seed, window_size, length, use_instrument=use_instrument, encode_section=encode_section)
		if instrument_program != None:
			midis.append(_network_output_to_midi(gen, instrument_program=instrument_program))
		else:
			midis.append(_network_output_to_midi(gen, instrument_name=instrument_name))
	return midis


# create a midi instrument using the one-hot encoding output of keras model.predict.
def _network_output_to_instrument(windows,
							instrument_program=0,
							allow_represses=False):
	# Create an Instrument instance
	instrument = pretty_midi.Instrument(program=instrument_program)

	cur_note = None  # an invalid note to start with
	cur_note_start = None
	clock = 0

	# Iterate over note names, which will be converted to note number later
	for step in windows:

		note_num = np.argmax(step) - 1

		# a note has changed
		if allow_represses or note_num != cur_note:

			# if a note has been played before and it wasn't a rest
			if cur_note is not None and cur_note >= 0:
				# add the last note, now that we have its end time
				note = pretty_midi.Note(velocity=127,
										pitch=int(cur_note),
										start=cur_note_start,
										end=clock)
				instrument.notes.append(note)

			# update the current note
			cur_note = note_num
			cur_note_start = clock

		# update the clock
		clock = clock + 1.0 / 4

	return instrument

# create a pretty midi file with a single instrument using the one-hot encoding
# output of keras model.predict.
def _network_output_to_midi(windows,
							instrument_name=None,
							instrument_program=None,
							allow_represses=False):
	# Create a PrettyMIDI object
	midi = pretty_midi.PrettyMIDI()
	# Create an Instrument instance

	if instrument_program is None and instrument_name is None:
		instrument_program = 0
	elif instrument_program is not None:
		pass
	elif instrument_name is not None:
		instrument_program = pretty_midi.instrument_name_to_program(instrument_name)
	instrument = _network_output_to_instrument(windows, instrument_program)

	# Add the instrument to the PrettyMIDI object
	midi.instruments.append(instrument)
	return midi


# Read instruments (map program id to instrument family)
instruments = defaultdict(lambda: 0)  # Default = 0 (piano)
families = []
family_instruments = []
with open('instruments.json') as json_file:
	data = json.load(json_file)
	for instrument in data:
		instrument_id = int(instrument['hexcode'], 16)
		if instrument['family'] in families:
			fam_id = families.index(instrument['family'])
		else:
			fam_id = len(families)
			family_instruments.append(instrument_id)
			families.append(instrument['family'])
		instruments[instrument_id] = fam_id


def get_family_id_by_instrument_normalized(instrument_id):
	return instruments[instrument_id] / len(families)


def get_family_instrument_by_normalized_class(normalized_class):
	fam_id = int(normalized_class * len(families))
	return family_instruments[fam_id]


# returns X, y data windows from all monophonic instrument
# tracks in a pretty midi file
def _windows_from_monophonic_instruments(midi, window_size, use_instrument=False, ignore_empty=False, encode_section=False):
	X, y = [], []
	for m in midi:
		if m is not None:
			melody_instruments = filter_monophonic(m.instruments, 1.0)
			for instrument in melody_instruments:
				if len(instrument.notes) > window_size:
					windows = _encode_sliding_windows(instrument, window_size)
					instrument_group = get_family_id_by_instrument_normalized(instrument.program)
					track_length = len(windows)
					for section, w in enumerate(windows):
						x_vals = w[0]
						if ignore_empty and np.min(w[0][:, 0]) == 1 and w[1][0] == 1:
							# Window only contains pauses and Y is also a pause.. ignore!
							continue
						if use_instrument:
							# Append instrument class to input (normalized to 0>1)
							x_vals = np.insert(x_vals, 0, instrument_group, axis=1)
						if encode_section:
							# Append track section to input (try to model intro, chorus, outro, etc)
							sections = [0] * 4
							active_section = int((section / track_length) * 4)
							sections[active_section] = 1
							section_matrix = np.array([sections,] * window_size)
							x_vals = np.concatenate((section_matrix, x_vals), axis=1)
						X.append(x_vals)
						y.append(w[1])
	return (np.asarray(X), np.asarray(y))


# returns X, y data windows from all tracks
def _windows_from_tracks(tracks, window_size, use_instrument=False, ignore_empty=False, encode_section=False):
	X, y = [], []
	for instrument in tracks:
		roll = instrument['roll']
		if len(roll) > window_size:
			windows = []
			for i in range(0, roll.shape[0] - window_size - 1):
				windows.append((roll[i:i + window_size], roll[i + window_size + 1]))
			instrument_group = instrument['instrument']
			track_length = len(windows)
			for section, w in enumerate(windows):
				x_vals = w[0]
				if ignore_empty and np.min(w[0][:, 0]) == 1 and w[1][0] == 1:
					# Window only contains pauses and Y is also a pause.. ignore!
					continue
				if use_instrument:
					# Append instrument class to input (normalized to 0>1)
					x_vals = np.insert(x_vals, 0, instrument_group, axis=1)
				if encode_section:
					# Append track section to input (try to model intro, chorus, outro, etc)
					sections = [0] * 4
					active_section = int((section / track_length) * 4)
					sections[active_section] = 1
					section_matrix = np.array([sections,] * window_size)
					x_vals = np.concatenate((section_matrix, x_vals), axis=1)
				X.append(x_vals)
				y.append(w[1])
	return (np.asarray(X), np.asarray(y))


# one-hot encode a sliding window of notes from a pretty midi instrument.
# expects pm_instrument to be monophonic.
def _encode_sliding_windows(pm_instrument, window_size):
	roll = get_instrument_roll(pm_instrument)
	windows = []
	for i in range(0, roll.shape[0] - window_size - 1):
		windows.append((roll[i:i + window_size], roll[i + window_size + 1]))
	return windows


# This approach uses the piano roll method, where each step in the sliding
# window represents a constant unit of time (fs=4, or 1 sec / 4 = 250ms).
# This allows us to encode rests.
def get_instrument_roll(pm_instrument):
	roll = np.copy(pm_instrument.get_piano_roll(fs=4).T)

	# trim beginning silence
	summed = np.sum(roll, axis=1)
	mask = (summed > 0).astype(float)
	roll = roll[np.argmax(mask):]

	# transform note velocities into 1s
	roll = (roll > 0).astype(float)

	# calculate the percentage of the events that are rests
	# s = np.sum(roll, axis=1)
	# num_silence = len(np.where(s == 0)[0])
	# print('{}/{} {:.2f} events are rests'.format(num_silence, len(roll), float(num_silence)/float(len(roll))))

	# append a feature: 1 to rests and 0 to notes
	rests = np.sum(roll, axis=1)
	rests = (rests != 1).astype(float)
	roll = np.insert(roll, 0, rests, axis=1)
	return roll