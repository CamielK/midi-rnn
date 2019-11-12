import os
from typing import List

import pretty_midi
import numpy as np
import librosa.display
from matplotlib import pyplot as plt
from pretty_midi import Instrument

import copy

data_dir = "midi-data"
output_dir = "midi-rs-clean"

midi_files = [os.path.join(data_dir, path) \
			  for path in os.listdir(data_dir) \
			  if '.mid' in path or '.midi' in path]

def save_cleaned_instrument(instrument: Instrument):
	"""
		Save the instrument as a standalone midi file in the cleaned directory
		This function also removes long pauses in the new tracks to compensate for instruments that only play in a small part of the original song
	"""
	try:
		# Create dummy
		midi = pretty_midi.PrettyMIDI(initial_tempo=80)
		midi.instruments.append(Instrument(program=instrument.program))

		# Adjust timing: Removes all pauses longer than 1 second
		max_pause = 1  # in seconds (float)
		original_notes: List[pretty_midi.Note] = pm.instruments[0].notes
		original_times = []
		new_times = []
		last_end = 0
		shift = 0
		for note in original_notes:
			original_times.append(note.start)

			# Check timings
			if note.start > last_end + max_pause:
				# Pause is too long! adjust timing by shifting left
				diff = note.start - last_end
				shift += diff - max_pause

			new_times.append(note.start-shift)
			last_end = note.end

		# Synthesize
		midi.adjust_times(original_times, new_times)
		midi.synthesize()

		# Save
		filename = f"file{file_id}-ins{i}-notes{len(midi.instruments[0].notes)}-pitch{len(midi.instruments[0].pitch_bends)}-controls{len(midi.instruments[0].control_changes)}"
		print(f"\t\tSaving instrument: {filename}")
		midi.write(f"{output_dir}/{filename}-ori_len{int(last_end)}-new_len{int(last_end-shift)}.mid")
	except Exception as e:
		print(f"Exception while cleaning instrument: {str(e)}")

# for k, file in enumerate(midi_files[:100]):
for k, file in enumerate(midi_files):
	print(f"Progress: {k}/{len(midi_files)} - Midi file: {file}")
	file_id = file[file.find("\\") + 1:-4]

	pm = pretty_midi.PrettyMIDI(file)
	pm.remove_invalid_notes()

	if pm.instruments[0].is_drum:
		print("Track is drum! Skipping")
		continue

	# CLEAN: Seperate instruments and adjust timing
	for i, instrument in enumerate(pm.instruments):
		if len(instrument.notes) > 1000 and not instrument.is_drum:
			# Save new instrument
			save_cleaned_instrument(instrument)

t = 2
