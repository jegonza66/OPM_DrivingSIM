import functions_analysis
import load
import setup
import paths
import matplotlib.pyplot as plt


#----- Path -----#
exp_info = setup.exp_info()

#----- Save data and display figures -----#
save_data = True
save_fig = True
display_figs = False
plot_individuals = True
if display_figs:
    plt.ion()
else:
    plt.ioff()

#-----  Parameters -----#
trial_params = {
                
                }

meg_params = {'chs_id': 'mag',
              'band_id': None,
              'data_type': 'ICA_annot'
              }

# TRF parameters
trf_params = {'input_features': ['fixation', 'saccade'],   # Select features (events)
              'standarize': True,
              'fit_power': False,
              'alpha': None,
              'tmin': -0.2,
              'tmax': 0.5,
              }
trf_params['baseline'] = (trf_params['tmin'], -0.05)

# Figure path
fig_path = paths.plots_path + (f"TRF_{meg_params['data_type']}/Band_{meg_params['band_id']}/{trf_params['input_features']}"
                               f"_{trf_params['tmin']}_{trf_params['tmax']}_bline{trf_params['baseline']}_alpha{trf_params['alpha']}_"
                               f"std{trf_params['standarize']}/{meg_params['chs_id']}/")

# Change path to include envelope power
if trf_params['fit_power']:
    fig_path = fig_path.replace(f"TRF_{meg_params['data_type']}", f"TRF_{meg_params['data_type']}_ENV")

# Save path
save_path = fig_path.replace(paths.plots_path, paths.save_path)

# Define Grand average variables
feature_evokeds = {}
for feature in trf_params['input_features']:
    feature_evokeds[feature] = []

# Iterate over subjects
for subject_id in exp_info.subjects_ids:
    trf_path = save_path
    trf_fname = f'TRF_{subject_id}.pkl'

    # Load subject
    subject = setup.subject(subject_id=subject_id)
    # Load MEG data
    meg_data = load.meg(subject_id=subject_id, meg_params=meg_params)

    try:
        # Load TRF
        rf = load.var(trf_path + trf_fname)
        print('Loaded Receptive Field')

    except:
        # Compute TRF for defined features
        rf = functions_analysis.compute_trf(subject=subject, meg_data=meg_data, trf_params=trf_params, meg_params=meg_params,
                                            save_data=save_data, trf_path=trf_path, trf_fname=trf_fname)

    # Get model coeficients as separate responses to each feature
    subj_evoked, feature_evokeds = functions_analysis.make_trf_evoked(subject=subject, rf=rf, meg_data=meg_data, evokeds=feature_evokeds,
                                                                      trf_params=trf_params, meg_params=meg_params,
                                                                      plot_individuals=plot_individuals, save_fig=save_fig, fig_path=fig_path)

fname = f"{feature}_GA_{meg_params['chs_id']}"
grand_avg = functions_analysis.trf_grand_average(feature_evokeds=feature_evokeds, trf_params=trf_params, trial_params=trial_params, meg_params=meg_params,
                                                 display_figs=display_figs, save_fig=save_fig, fig_path=fig_path)
