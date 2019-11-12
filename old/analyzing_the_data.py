import os
from typing import List

import pretty_midi
import numpy as np
import librosa.display
from matplotlib import pyplot as plt
from pretty_midi import Instrument

import copy

data_dir = "midi-data"
# data_dir = "code/data/midi"
# data_dir = "midi-rs-clean"


midi_files = [os.path.join(data_dir, path) \
			  for path in os.listdir(data_dir) \
			  if '.mid' in path or '.midi' in path]


def plot_piano_roll(pm, start_pitch, end_pitch, fs=100):
	# Use librosa's specshow function for displaying the piano roll
	librosa.display.specshow(pm.get_piano_roll(fs)[start_pitch:end_pitch],
							 hop_length=1, sr=fs, x_axis='time', y_axis='cqt_note',
							 fmin=pretty_midi.note_number_to_hz(start_pitch))


# midi_files = [
# 	"midi-rs-clean/file1001-ins2-notes1164-pitch0-controls174-ori_len155-new_len41.mid",
# 	"midi-rs-clean/file1000-ins5-notes1768-pitch0-controls721-ori_len312-new_len120.mid",
# 	"midi-rs-clean/file1000-ins3-notes4248-pitch0-controls50-ori_len312-new_len120.mid",
# 	"midi-rs-clean/file1000-ins10-notes1472-pitch0-controls4-ori_len312-new_len120.mid",
# ]

stats = {
	'name': [],
	'num_notes': [],
	'tempo': [],
	'pitch_bends': [],
	'control_changes': [],
	'sign_changes': [],
	'pitch_hist': [],
	'programs': []
}

# for k, file in enumerate(midi_files[:100]):
for k, file in enumerate(midi_files):
	print(f"Progress: {k}/{len(midi_files)} - Midi file: {file}")
	file_id = file[file.find("\\") + 1:-4]

	pm = pretty_midi.PrettyMIDI(file)

	# plt.figure(figsize=(12, 4))
	# plt.title(f"Midi file: {file}")
	# plot_piano_roll(pm, 24, 84)
	# plt.show()

	for instrument in pm.instruments:
		stats['programs'].append(instrument.program)

	# stats['name'].append(file)
	# stats['num_notes'].append(len(pm.instruments[0].notes))
	# stats['tempo'].append(pm.estimate_tempo())
	# stats['pitch_bends'].append(len(pm.instruments[0].pitch_bends))
	# stats['control_changes'].append(pm.instruments[0].control_changes)
	# stats['sign_changes'].append(len(pm.time_signature_changes))
	# stats['pitch_hist'].append(pm.get_pitch_class_histogram())
	# print('\t\tThere are {} time signature changes'.format(len(pm.time_signature_changes)))
	# print('\t\tThe tempo estimate is {}'.format(pm.estimate_tempo()))
	# print('\t\tInstrument 0 has {} notes'.format(len(pm.instruments[0].notes)))
	# print('\t\tInstrument 0 has {} pitch bends'.format(len(pm.instruments[0].pitch_bends)))
	# print('\t\tInstrument 0 has {} control changes'.format(len(pm.instruments[0].control_changes)))

	t=2

t = 2
