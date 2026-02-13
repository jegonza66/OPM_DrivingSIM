import paths
import mne
import pandas as pd
import pathlib
import os


class exp_info:
    """
    Class containing the experiment information.

    Attributes
    ----------
    ctf_path : str
        Path to the MEG data.
    et_path : str
        Path to the Eye-Tracker data.
    mri_path : str
        Path to the MRI data.
    opt_path : str
        Path to the Digitalization data.
    subjects_ids : list
        List of subject IDs.
    bad_channels : dict
        Dictionary of subject IDs and their corresponding bad channels.
    screen_distance : dict
        Dictionary of subject IDs and their distance to the screen during the experiment.
    screen_size : dict
        Dictionary of subject IDs and their screen size.
    group : dict
        Dictionary of subject IDs and their group (balanced or counterbalanced).
    et_channel_names : dict
        Dictionary of tracked eyes and their corresponding channel names.
    tracked_eye : dict
        Dictionary of subject IDs and their tracked eye.
    no_pupil_subjects : list
        List of subjects with missing pupil data.
    trig_ch : str
        Trigger channel name.
    button_ch : str
        Buttons channel name.
    buttons_ch_map : dict
        Map of button values to colors.
    buttons_pc_map : dict
        Map of button values to colors (PC version).
    DAC_delay : int
        DAC delay in milliseconds.
    noise_recordings : list
        List of background noise recordings date IDs.
    empty_room_recording : dict
        Dictionary of subject IDs and their associated background noise.
    background_line_noise_freqs : dict
        Dictionary of noise recording dates and their line noise frequencies.
    line_noise_freqs : dict
        Dictionary of subject IDs and their line noise frequencies.
    """

    def __init__(self):
        # Define opm data path and files path
        self.opm_path = paths.opm_path
        self.et_path = paths.et_path
        self.mri_path = paths.mri_path
        self.opt_path = paths.opt_path

        # Select subject
        self.subjects_ids = [
            # '17976',
            '17643',
            '10925'
        ]

        # Subjects bad channels
        self.bad_channels = {
                             }

        # # Taken from Driving Experiment info v0_1
        # self.exp_times = {'17976': {'cf_start': 59, 'cf_end': 172, 'da_start': 172, 'da_end': 535},
        #                   '17643': {'cf_start': 49, 'cf_end': 161, 'da_start': 161, 'da_end': 526},
        #                   '10925': {'cf_start': 55, 'cf_end': 168, 'da_start': 161, 'da_end': 546}
        #                   }

        # Dataframe containing DA symbols start times for each subject
        # self.da_times = {key: pd.read_csv(paths.bh_path + f'DA_EVENT_TIME_1.csv')[key] for key in self.subjects_ids}

        # Dataframe containing DA symbols start times for each subject
        # master_df = pd.read_csv(paths.exp_path + f'master_df.csv')
        # self.master_df = {key: master_df.loc[master_df['part_id'] == int(key)] for key in self.subjects_ids}

        # Distance to the screen during the experiment (Fake info)
        self.screen_distance = {'17976': 62,
                                '17643': 60,
                                '10925': 60
                                }

        # Screen width
        self.screen_size = 56.8

        # Screen resoltion
        self.screen_res = [1920, 1080]

        # Video resolution
        self.video_res = [1920, 1080]

        # Mirrors size
        self.mirrors_size = [250, 200]

        # Mirrors center
        self.left_mirror_center = [250, 900]

        self.left_mirror_px = {'x': [self.left_mirror_center[0] - self.mirrors_size[0]/2,
                                     self.left_mirror_center[0] + self.mirrors_size[0]/2],
                               'y': [self.left_mirror_center[1] - self.mirrors_size[1]/2,
                                     self.left_mirror_center[1] + self.mirrors_size[1]/2]}

        # Right mirror pixels range
        self.right_mirror_px = {'x': [self.screen_res[0] - self.left_mirror_center[0] - self.mirrors_size[0]/2,
                                      self.screen_res[0] - self.left_mirror_center[0] + self.mirrors_size[0]/2],
                                'y': [self.left_mirror_center[1] - self.mirrors_size[1]/2,
                                      self.left_mirror_center[1] + self.mirrors_size[1]/2]}

        # Left mirror pixels range
        # self.left_mirror_px = {'x':[(self.screen_res[0] - self.video_res[0])/2,
        #                             (self.screen_res[0] - self.video_res[0])/2 + self.mirrors_size_px[0]],
        #                        'y':[(self.screen_res[1] + self.video_res[1])/2 - self.mirrors_size_px[1],
        #                             (self.screen_res[1] + self.video_res[1])/2]}
        #
        # # Right mirror pixels range
        # self.right_mirror_px = {'x':[(self.screen_res[0] + self.video_res[0])/2 - self.mirrors_size_px[0],
        #                             (self.screen_res[0] + self.video_res[0])/2],
        #                        'y':[(self.screen_res[1] + self.video_res[1])/2 - self.mirrors_size_px[1],
        #                             (self.screen_res[1] + self.video_res[1])/2]}

        # Subjects groups (Fake info)
        self.group = {'17976': 'balanced',
                      '17643': 'balanced',
                      '10925': 'balanced'
                      }

        # Duration of the DA in seconds
        self.DA_duration = 4.5

        # Tracked eye (Fake info)
        self.tracked_eye = {'17976': 'left',
                            '17643': 'left',
                            '10925': 'right'
                            }

        # ET channels name [Gaze x, Gaze y, Pupils] (Fake info)
        self.et_channel_names = {'17976': ['meg_x', 'meg_y', 'meg_pupil'],
                                 '17643': ['meg_x', 'meg_y', 'meg_pupil'],
                                 '10925': ['meg_x', 'meg_y', 'meg_pupil']
                                 }

        # Trigger channel name (Fake info)
        self.trig_ch = 'UPPT002'
        self.alt_trig_ch = 'UPPT001'

        # Buttons channel name (Fake info)
        self.button_ch = 'UPPT001'

        # Buttons values to colors map
        self.buttons_ch_map = {1: 'blue', 2: 'yellow', 4: 'green', 8: 'red', 'blue': 1, 'yellow': 2, 'green': 4, 'red': 8}
        self.buttons_pc_map = {1: 'blue', 2: 'yellow', 3: 'green', 4: 'red', 'blue': 1, 'yellow': 2, 'green': 3, 'red': 4}

        # DAC delay (in ms)
        self.DAC_delay = 10

        # Notch filter line noise frequencies (Fake info)
        self.line_noise_freqs = {'17976': (50, 57, 100, 109, 150, 200, 250, 300),
                                 '17643': (50, 57, 100, 109, 150, 200, 250, 300),
                                 '10925': (50, 57, 100, 109, 150, 200, 250, 300),
                                 }


class analysis_parameters:
    """
    Class containing the analysis parameters.

    Attributes
    ----------
    ctf_path: str
        Path to the MEG data.
    et_path: str
        Path to the Eye-Tracker data.
    mri_path: str
        Path to the MRI data.
    opt_path: str
        Path to the Digitalization data.
    subjects_ids: list
        List of subject's id.
    subjects_bad_channels: list
        List of subject's bad channels.
    subjects_groups: list
        List of subject's group
    missing_bh_subjects: list
        List of subject's ids missing behavioural data.
    trials_loop_subjects: list
        List of subject;s ids for subjects that took the firts version of the experiment.
    """

    def __init__(self):
        # Samples drop at begining of missing pupils signal
        self.start_interval_samples = 24

        # Samples drop at end of missing pupils signal
        self.end_interval_samples = 24

        # Pupil size threshold to consider missing signal (Fake info)
        self.pupil_thresh = {'17976': -2.6,
                             '17643': -2.6,
                             '10925': -2.6,
                             }

        # Et samples shift for ET-MEG alignment
        self.et_samples_shift = {}

        # Trial reject parameter based on MEG peak to peak amplitude (Fake info)
        self.reject_amp = {'17976': 5e-12,
                           '17643': 5e-12,
                           '10925': 5e-12
                           }


class subject():
    """
    Class containing subjects data and analysis parameters.

    Parameters
    ----------
    subject: {'int', 'str'}, default=None
        Subject id (str) or number (int). If None, takes the first subject.

    Attributes
    ----------
    bad_channels: list
        List of bad channels.
    ctf_path: str
        Path to the MEG data.
    et_path: str
        Path to the Eye-Tracker data.
    mri_path: str
        Path to the MRI data.
    opt_path: str
        Path to the Digitalization data.
    subject_id: str
        Subject id.
    """

    def __init__(self, subject_id=None):

        # Select 1st subject by default
        if subject_id == None:
            self.subject_id = exp_info().subjects_ids[0]
        # Select subject by index
        elif type(subject_id) == int:
            self.subject_id = exp_info().subjects_ids[subject_id]
        # Select subject by id
        elif type(subject_id) == str and (subject_id in exp_info().subjects_ids):
            self.subject_id = subject_id
        else:
            print('Subject not found')

        # Get attributes from experiment info
        exp_info_att = exp_info().__dict__.keys()
        for general_att in exp_info_att:
            att = getattr(exp_info(), general_att)
            if type(att) == dict:
                try:
                    # If subject_id in dictionary keys, get attribute
                    att_value = att[self.subject_id]
                    setattr(self, general_att, att_value)
                except:
                    pass

        # Get preprocessing and general configurations
        self.params = self.params(params=analysis_parameters(), subject_id=self.subject_id)

    # Subject's parameters and configuration
    class params:

        def __init__(self, params, subject_id):
            # Get attributes and get data for corresponding subject
            attributes = params.__dict__.keys()

            # Iterate over attributes and get data for corresponding subject
            for att_name in attributes:
                att = getattr(params, att_name)
                if type(att) == dict:
                    try:
                        # If subject_id in dictionary keys, get attribute, else pass
                        att_value = att[subject_id]
                        setattr(self, att_name, att_value)
                    except:
                        pass
                else:
                    # If attribute is general for all subjects, get attribute
                    att_value = att
                    setattr(self, att_name, att_value)

    # Raw MEG data
    def load_raw_meg_data(self):
        """
        MEG data for parent subject as Raw instance of MNE.
        """

        print('\nLoading Raw MEG data')
        # Get subject path
        subj_path = pathlib.Path(os.path.join(paths.opm_path, self.subject_id))
        ds_files = list(subj_path.glob('*{}*.ds'.format(self.subject_id)))
        ds_files.sort()

        # Load sesions
        # If more than 1 session concatenate all data to one raw data
        if len(ds_files) > 1:
            raws_list = []
            for i in range(len(ds_files)):
                raw = mne.io.read_raw_ctf(ds_files[i], system_clock='ignore')
                raws_list.append(raw)
            # MEG data structure
            raw = mne.io.concatenate_raws(raws_list, on_mismatch='ignore')

            # Set dev <-> head transformation from optimal head localization
            raw.info['dev_head_t'] = raws_list[self.params.general.head_loc_idx].info['dev_head_t']

        # If only one session return that session as whole raw data
        elif len(ds_files) == 1:
            raw = mne.io.read_raw_ctf(ds_files[0], system_clock='ignore')

        # Missing data
        else:
            raise ValueError('No .ds files found in subject directory: {}'.format(subj_path))

        return raw

    # MEG data
    def load_preproc_meg_data(self, preload=False):
        """
        Preprocessed MEG data for parent subject as raw instance of MNE.
        """

        # Subject preprocessed data path
        file_path = pathlib.Path(os.path.join(paths.preproc_path, self.subject_id, f'Subject_{self.subject_id}_meg.fif'))

        # Try to load preprocessed data
        try:
            print('\nLoading Preprocessed MEG data')
            meg_data = mne.io.read_raw_fif(file_path, preload=preload)
        except:
            raise ValueError(f'No previous preprocessed data found for subject {self.subject_id}')

        return meg_data

    # MEG data
    def load_processed_meg_data(self, preload=False):
        """
        Processed MEG data for parent subject as raw instance of MNE.
        """

        # Subject preprocessed data path
        file_path = pathlib.Path(os.path.join(paths.processed_path, self.subject_id, f'Subject_{self.subject_id}_meg.fif'))

        # Try to load preprocessed data
        try:
            print('\nLoading Preprocessed MEG data')
            meg_data = mne.io.read_raw_fif(file_path, preload=preload)
        except:
            raise ValueError(f'No previous preprocessed data found for subject {self.subject_id}')

        return meg_data

    # ICA MEG data
    def load_ica_meg_data(self, preload=False):
        """
        ICA MEG data for parent subject as Raw instance of MNE.
        """

        # Subject ICA data path
        file_path = pathlib.Path(os.path.join(paths.ica_path, self.subject_id, f'Subject_{self.subject_id}_ICA.fif'))

        # Try to load ica data
        try:
            print(f'Loading ICA data for subject {self.subject_id}')
            # Load data
            ica_data = mne.io.read_raw_fif(file_path, preload=preload)

        except:
            raise ValueError(f'No previous ica data found for subject {self.subject_id}')

        return ica_data


   # Behavioural data
    def load_raw_bh_data(self):
        """
        Behavioural data for parent subject as pandas DataFrames.
        """
        # Get subject path
        raise ('Update paths to use csv within ET data.')
        subj_path = pathlib.Path(os.path.join(paths.eh_path, self.subject_id))
        bh_file = list(subj_path.glob('*.csv'.format(self.subject_id)))[0]

        # Load DataFrame
        df = pd.read_csv(bh_file)

        return df

    # Fixations data
    def fixations(self):
        """
        Fixations data for parent subject as pandas DataFrame.
        Loads the fixations CSV file saved during preprocessing.

        Returns
        -------
        fixations : pandas.DataFrame
            DataFrame containing fixation events with columns like onset, duration,
            amplitude, mean_x, mean_y, pupil size, etc.
        """

        # Subject fixations data path
        file_path = pathlib.Path(os.path.join(paths.processed_path, self.subject_id, 'fixations.csv'))

        # Try to load fixations data
        try:
            print(f'Loading fixations DataFrame for subject {self.subject_id}')
            fixations_data = pd.read_csv(file_path)
        except FileNotFoundError:
            raise ValueError(f'No fixations DataFrame found for subject {self.subject_id}. '
                           f'Expected file: {file_path}')
        except Exception as e:
            raise ValueError(f'Error loading fixations DataFrame for subject {self.subject_id}: {e}')

        return fixations_data

    # Saccades data
    def saccades(self):
        """
        Saccades data for parent subject as pandas DataFrame.
        Loads the saccades CSV file saved during preprocessing.

        Returns
        -------
        saccades : pandas.DataFrame
            DataFrame containing saccade events with columns like onset, duration,
            amplitude, peak_vel, n_sac, delay, deg, dir, etc.
        """

        # Subject saccades data path
        file_path = pathlib.Path(os.path.join(paths.processed_path, self.subject_id, 'saccades.csv'))

        # Try to load saccades data
        try:
            print(f'Loading saccades DataFrame for subject {self.subject_id}')
            saccades_data = pd.read_csv(file_path)
        except FileNotFoundError:
            raise ValueError(f'No saccades DataFrame found for subject {self.subject_id}. '
                           f'Expected file: {file_path}')
        except Exception as e:
            raise ValueError(f'Error loading saccades DataFrame for subject {self.subject_id}: {e}')

        return saccades_data

    # Pursuits data
    def pursuits(self):
        """
        Pursuits data for parent subject as pandas DataFrame.
        Loads the pursuits CSV file saved during preprocessing.

        Returns
        -------
        pursuits : pandas.DataFrame
            DataFrame containing pursuit events with columns like onset, duration,
            amplitude, etc.
        """

        # Subject pursuits data path
        file_path = pathlib.Path(os.path.join(paths.processed_path, self.subject_id, 'pursuits.csv'))

        # Try to load pursuits data
        try:
            print(f'Loading pursuits DataFrame for subject {self.subject_id}')
            pursuits_data = pd.read_csv(file_path)
        except FileNotFoundError:
            raise ValueError(f'No pursuits DataFrame found for subject {self.subject_id}. '
                           f'Expected file: {file_path}')
        except Exception as e:
            raise ValueError(f'Error loading pursuits DataFrame for subject {self.subject_id}: {e}')

        return pursuits_data