"""Master table of experiment event times for every participant, in MEG time.

Columns
-------
subject        participant id
event          phase / marker name (CF, DA, Audio, DA_stim, drive, inst*, em_*, ...)
label          trigger label for DA stimuli (LL/1, LR/1, RL/4, RR/4), else ''
index          1-based stimulus number for DA_stim, else ''
onset_s        0-based seconds from the first sample of the processed raw
offset_s       0-based end (NaN for point markers)
onset_abs_s    same, in the raw's annotation time base (includes first_time)
offset_abs_s
duration_s
first_time_s   raw.first_time of that participant (abs = 0-based + first_time)
source         'csv+offset' (simulator clock realigned), 'trigger' (raw annotation)

Phases CF / Audio come from the behavioural CSVs realigned to the MEG clock via
the DA hardware triggers (functions_analysis.get_experiment_phase_times); DA and
all point markers come straight from the raw annotations.
"""
import os
import numpy as np
import pandas as pd

import paths
import setup
import load
import functions_analysis

OUT_FILE = os.path.join(paths.save_path, 'MEG_event_times_master.csv')

# Point markers to export from the raw annotations (besides the DA triggers)
MARKERS = ('inst2', 'em_blink', 'em_horz', 'em_vert', 'em_horz2', 'em_vert2', 'em_dyn',
           'inst3', 'gas', 'brake', 'right_steer', 'left_steer', 'inst4', 'drive')

rows = []
for subject_id in setup.exp_info().subjects_ids:
    print(f'\n>>> {subject_id}')
    meg_data = load.meg(subject_id=subject_id, meg_params={'data_type': 'processed'})
    t0 = meg_data.first_time
    rec_end = meg_data.times[-1]

    def add(event, on, off=np.nan, label='', index='', source='trigger'):
        rows.append(dict(subject=subject_id, event=event, label=label, index=index,
                         onset_s=on, offset_s=off, onset_abs_s=on + t0, offset_abs_s=off + t0,
                         duration_s=off - on, first_time_s=t0, source=source))

    phases = functions_analysis.get_experiment_phase_times(subject_id, meg_data)
    add('CF', *phases['CF'], source='csv+offset')
    add('DA', *phases['DA'], source='trigger')
    add('Audio', *phases['Audio'], source='csv+offset')
    if phases['CF'][1] > rec_end:
        print(f'    CF end ({phases["CF"][1]:.1f} s) is past the recording end ({rec_end:.1f} s)')

    da_onsets, da_labels = functions_analysis.get_da_stimulus_onsets(meg_data)
    for k, (on, lab) in enumerate(zip(da_onsets, da_labels), start=1):
        add('DA_stim', on, on + setup.exp_info().DA_duration, label=lab, index=k)

    desc = np.asarray(meg_data.annotations.description)
    onsets = meg_data.annotations.onset - t0
    for marker in MARKERS:
        hits = np.flatnonzero(np.char.strip(desc.astype(str)) == marker)  # ' left_steer' has a stray space
        if len(hits) == 0:
            print(f'    marker {marker!r} not found')
            continue
        add(marker, onsets[hits[0]])

    add('recording', 0.0, rec_end)

df = pd.DataFrame(rows)
df.to_csv(OUT_FILE, index=False, float_format='%.3f')
print(f'\n>>> Saved {len(df)} rows to {OUT_FILE}')

# Compact overview: one line per subject with phase boundaries
overview = (df[df.event.isin(['drive', 'CF', 'DA', 'Audio', 'recording'])]
            .pivot(index='subject', columns='event', values='onset_s')
            .join(df[df.event.isin(['CF', 'DA', 'Audio', 'recording'])]
                  .pivot(index='subject', columns='event', values='offset_s'), rsuffix='_end'))
print(overview[['drive', 'CF', 'CF_end', 'DA', 'DA_end', 'Audio', 'Audio_end', 'recording_end']].round(1).to_string())
