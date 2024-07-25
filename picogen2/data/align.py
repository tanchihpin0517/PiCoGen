import copy
import json
import os
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
import pretty_midi
import pyrubberband as pyrb
from synctoolbox.dtw.mrmsdtw import sync_via_mrmsdtw
from synctoolbox.dtw.utils import (
    compute_optimal_chroma_shift,
    make_path_strictly_monotonic,
    shift_chroma_vectors,
)
from synctoolbox.feature.chroma import (
    pitch_to_chroma,
    quantize_chroma,
    quantized_chroma_to_CENS,
)
from synctoolbox.feature.dlnco import pitch_onset_features_to_DLNCO
from synctoolbox.feature.pitch import audio_to_pitch_features
from synctoolbox.feature.pitch_onset import audio_to_pitch_onset_features
from synctoolbox.feature.utils import estimate_tuning

from ..utils import normalize

Fs = 44100
feature_rate = 50
step_weights = np.array([1.5, 1.5, 2.0])
threshold_rec = 10**6


def save_delayed_song(
    song_audio_file: Path,
    piano_midi_file: Path,
    align_info_file: Path,
):
    song_audio, _ = librosa.load(str(song_audio_file), sr=Fs)
    piano_midi = pretty_midi.PrettyMIDI(str(piano_midi_file))

    rd = get_aligned_results(song_audio=song_audio, piano_midi=piano_midi)

    song_pitch_shifted = rd["song_pitch_shifted"]
    midi_warped_pm = rd["midi_warped_pm"]
    pitch_shift_for_song_audio = rd["pitch_shift_for_song_audio"]
    tuning_offset_song = rd["tuning_offset_song"]
    tuning_offset_piano = rd["tuning_offset_piano"]
    time_map_second = rd["time_map_second"]

    align_info = dict(
        time_map=time_map_second,
        pitch_shift=pitch_shift_for_song_audio.item(),
        tuning_offset_song=tuning_offset_song.item(),
        tuning_offset_piano=tuning_offset_piano.item(),
    )
    align_info_file.write_text(json.dumps(align_info, indent=4))


def _fast_fluidsynth(midi, fs):
    """
    Faster fluidsynth synthesis using the command-line program
    instead of pyfluidsynth.
    Parameters
    ----------
    - m : pretty_midi.PrettyMIDI
        Pretty MIDI object
    - fs : int
        Sampling rate
    Returns
    -------
    - midi_audio : np.ndarray
        Synthesized audio, sampled at fs
    """
    # check fluidsynth is installed
    try:
        subprocess.check_output(["fluidsynth", "-h"])
    except FileNotFoundError:
        raise FileNotFoundError("fluidsynth is not installed. Please install it.")

    # Write out temp mid file
    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav")
    temp_mid = tempfile.NamedTemporaryFile(suffix=".mid")
    midi.write(temp_mid.name)

    # Get path to default pretty_midi SF2
    sf2_path = os.path.join(os.path.dirname(pretty_midi.__file__), pretty_midi.DEFAULT_SF2)
    # Make system call to fluidsynth
    with open(os.devnull, "w") as devnull:
        subprocess.check_output(
            ["fluidsynth", "-F", temp_wav.name, "-r", str(fs), sf2_path, temp_mid.name],
            stderr=devnull,
        )

    # Load from temp wav file
    audio, _ = librosa.load(temp_wav.name, sr=fs)
    # Occasionally, fluidsynth pads a lot of silence on the end, so here we
    # crop to the length of the midi object
    audio = audio[: int(midi.get_end_time() * fs)]

    return audio


def get_aligned_results(song_audio, piano_midi):
    DFs = 22050

    song_audio = normalize(song_audio)

    song_audio_downsampled = librosa.resample(song_audio, orig_sr=Fs, target_sr=DFs)
    piano_audio_downsampled = _fast_fluidsynth(piano_midi, DFs)

    # The reason for estimating tuning ::
    # https://www.audiolabs-erlangen.de/resources/MIR/FMP/C3/C3S1_TranspositionTuning.html
    tuning_offset_1 = estimate_tuning(song_audio_downsampled, DFs)
    tuning_offset_2 = estimate_tuning(piano_audio_downsampled, DFs)

    # DLNCO features (Sebastian Ewert, Meinard Müller, and Peter Grosche: High Resolution Audio Synchronization Using Chroma Onset Features, In Proceedings of IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP): 1869–1872, 2009.):
    # helpful to increase synchronization accuracy, especially for music with clear onsets.

    # Quantized and smoothed chroma : CENS features
    # Because, MrMsDTW Requires CENS.
    f_chroma_quantized_1, f_DLNCO_1 = get_features_from_audio(
        song_audio_downsampled, tuning_offset_1
    )
    f_chroma_quantized_2, f_DLNCO_2 = get_features_from_audio(
        piano_audio_downsampled, tuning_offset_2
    )

    # Shift chroma vectors :
    # Otherwise, different keys of two audio leads to degradation of alignment.
    opt_chroma_shift = compute_optimal_chroma_shift(
        quantized_chroma_to_CENS(f_chroma_quantized_1, 201, 50, feature_rate)[0],
        quantized_chroma_to_CENS(f_chroma_quantized_2, 201, 50, feature_rate)[0],
    )
    f_chroma_quantized_2 = shift_chroma_vectors(f_chroma_quantized_2, opt_chroma_shift)
    f_DLNCO_2 = shift_chroma_vectors(f_DLNCO_2, opt_chroma_shift)

    wp = sync_via_mrmsdtw(
        f_chroma1=f_chroma_quantized_1,
        f_onset1=f_DLNCO_1,
        f_chroma2=f_chroma_quantized_2,
        f_onset2=f_DLNCO_2,
        input_feature_rate=feature_rate,
        step_weights=step_weights,
        threshold_rec=threshold_rec,
        verbose=False,
    )

    wp = make_path_strictly_monotonic(wp)
    pitch_shift_for_song_audio = -opt_chroma_shift % 12
    if pitch_shift_for_song_audio > 6:
        pitch_shift_for_song_audio -= 12

    if pitch_shift_for_song_audio != 0:
        song_audio_shifted = pyrb.pitch_shift(song_audio, Fs, pitch_shift_for_song_audio)
    else:
        song_audio_shifted = song_audio

    time_map_second = wp / feature_rate
    piano_time = time_map_second[1]
    song_time = time_map_second[0]

    midi_pm_warped = copy.deepcopy(piano_midi)
    midi_pm_warped = simple_adjust_times(midi_pm_warped, piano_time, song_time)

    song_audio_shifted = normalize(song_audio_shifted)

    rd = dict(
        song_pitch_shifted=song_audio_shifted,
        midi_warped_pm=midi_pm_warped,
        pitch_shift_for_song_audio=pitch_shift_for_song_audio,
        tuning_offset_song=tuning_offset_1,
        tuning_offset_piano=tuning_offset_2,
        time_map_second={"piano": piano_time.tolist(), "song": song_time.tolist()},
    )
    return rd


def simple_adjust_times(pm, original_times, new_times):
    """
    most of these codes are from original pretty_midi
    https://github.com/craffel/pretty-midi/blob/main/pretty_midi/pretty_midi.py
    """
    for instrument in pm.instruments:
        instrument.notes = [
            copy.deepcopy(note)
            for note in instrument.notes
            if note.start >= original_times[0] and note.end <= original_times[-1]
        ]
    # Get array of note-on locations and correct them
    note_ons = np.array([note.start for instrument in pm.instruments for note in instrument.notes])
    adjusted_note_ons = np.interp(note_ons, original_times, new_times)
    # Same for note-offs
    note_offs = np.array([note.end for instrument in pm.instruments for note in instrument.notes])
    adjusted_note_offs = np.interp(note_offs, original_times, new_times)
    # Correct notes
    for n, note in enumerate([note for instrument in pm.instruments for note in instrument.notes]):
        note.start = (adjusted_note_ons[n] > 0) * adjusted_note_ons[n]
        note.end = (adjusted_note_offs[n] > 0) * adjusted_note_offs[n]
    # After performing alignment, some notes may have an end time which is
    # on or before the start time.  Remove these!
    pm.remove_invalid_notes()

    def adjust_events(event_getter):
        """This function calls event_getter with each instrument as the
        sole argument and adjusts the events which are returned."""
        # Sort the events by time
        for instrument in pm.instruments:
            event_getter(instrument).sort(key=lambda e: e.time)
        # Correct the events by interpolating
        event_times = np.array(
            [event.time for instrument in pm.instruments for event in event_getter(instrument)]
        )
        adjusted_event_times = np.interp(event_times, original_times, new_times)
        for n, event in enumerate(
            [event for instrument in pm.instruments for event in event_getter(instrument)]
        ):
            event.time = adjusted_event_times[n]
        for instrument in pm.instruments:
            # We want to keep only the final event which has time ==
            # new_times[0]
            valid_events = [
                event for event in event_getter(instrument) if event.time == new_times[0]
            ]
            if valid_events:
                valid_events = valid_events[-1:]
            # Otherwise only keep events within the new set of times
            valid_events.extend(
                event
                for event in event_getter(instrument)
                if event.time > new_times[0] and event.time < new_times[-1]
            )
            event_getter(instrument)[:] = valid_events

    # Correct pitch bends and control changes
    adjust_events(lambda i: i.pitch_bends)
    adjust_events(lambda i: i.control_changes)

    return pm


def get_features_from_audio(audio, tuning_offset, visualize=False):
    f_pitch = audio_to_pitch_features(
        f_audio=audio,
        Fs=Fs,
        tuning_offset=tuning_offset,
        feature_rate=feature_rate,
        verbose=visualize,
    )
    f_chroma = pitch_to_chroma(f_pitch=f_pitch)
    f_chroma_quantized = quantize_chroma(f_chroma=f_chroma)

    f_pitch_onset = audio_to_pitch_onset_features(
        f_audio=audio, Fs=Fs, tuning_offset=tuning_offset, verbose=visualize
    )
    f_DLNCO = pitch_onset_features_to_DLNCO(
        f_peaks=f_pitch_onset,
        feature_rate=feature_rate,
        feature_sequence_length=f_chroma_quantized.shape[1],
        visualize=visualize,
    )
    return f_chroma_quantized, f_DLNCO
