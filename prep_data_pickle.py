"""
	Prepare the midi windows to increase load time during training
	If the dataset is prepared, training will not have to read from disk and be much faster
"""
import os
import pickle
import random

from datetime import datetime
from pretty_midi import pretty_midi

import utils

# Params
data_dir = "data"
target = "pickle-data"
window_size = 20

midi_files = [os.path.join(data_dir, path) \
			  for path in os.listdir(data_dir) \
			  if '.mid' in path or '.midi' in path]
random.shuffle(midi_files)

total_events = 0
all_tracks = []
for i, path in enumerate(midi_files):
	print(f"Progress: {i}/{len(midi_files)}. total tracks: {len(all_tracks)}. total events: {total_events}")
	# Load midi
	midi = None
	try:
		midi = pretty_midi.PrettyMIDI(path)
		midi.remove_invalid_notes()
	except Exception as e:
		print(f"Skipping {path}, error: {e}")
		continue

	# Get tracks
	tracks = []
	melody_instruments = utils.filter_monophonic(midi.instruments, 1.0)
	for instrument in melody_instruments:
		if len(instrument.notes) > window_size:
			roll = utils.get_instrument_roll(instrument)
			if len(roll) > 0:
				instrument_group = utils.get_family_id_by_instrument_normalized(instrument.program)
				tracks.append({
					'roll': roll, 'instrument': instrument_group
				})
				total_events += len(roll)

	all_tracks += tracks

	del midi

print(f"Found a total of {len(all_tracks)} usable instrument tracks with a total of {total_events} events.")

# Dump
time = datetime.now().strftime("%Y%m%d_%H%M%S")
with open(f"{target}/dataset_{time}.pkl", 'wb') as f:
	pickle.dump(all_tracks, f)

t=2