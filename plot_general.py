import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import paths
import save
import functions_general
import functions_analysis
import mne
import mne_connectivity
from nilearn import plotting
from itertools import compress
import pandas as pd
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
import netplotbrain


save_path = paths.save_path
plot_path = paths.plots_path


def epochs(subject, epochs, picks, order=None, overlay=None, combine='mean', sigma=5, group_by=None, cmap='bwr',
           vmin=None, vmax=None, display_figs=True, save_fig=None, fig_path=None, fname=None):

    if save_fig and (not fname or not fig_path):
        raise ValueError('Please provide path and filename to save figure. Else, set save_fig to false.')

    fig_ep = epochs.plot_image(picks=picks, order=order, sigma=sigma, cmap=cmap, overlay_times=overlay, combine=combine,
                               vmin=vmin, vmax=vmax, title=subject.subject_id, show=display_figs)

    # Save figure
    if save_fig:
        if len(fig_ep) == 1:
            fig = fig_ep[0]
            save.fig(fig=fig, path=fig_path, fname=fname)
        else:
            for i in range(len(fig_ep)):
                fig = fig_ep[i]
                group = group_by.keys()[i]
                fname += f'{group}'
                save.fig(fig=fig, path=fig_path, fname=fname)


def evoked(evoked_meg, fig=None, axes=None, plot_xlim='tight', plot_ylim=None, display_figs=False, save_fig=True, fig_path=None, fname=None):
    '''
    Plot evoked response with mne.Evoked.plot() method. Option to plot gaze data on subplot.

    :param evoked_meg: Evoked with picked mag channels
    :param fig: Optional. Figure instance
    :param axes: Optional. Axes instance (if figure provided)
    :param plot_xlim: tuple. x limits for evoked and gaze plot
    :param plot_ylim: dict. Possible keys: meg, mag, eeg, misc...
    :param display_figs: bool. Whether to show figures or not
    :param save_fig: bool. Whether to save figures. Must provide save_path and figure name.
    :param fig_path: string. Optional. Path to save figure if save_fig true
    :param fname: string. Optional. Filename if save_fig is True

    :return: None
    '''

    # Sanity check
    if save_fig and (not fname or not fig_path):
        raise ValueError('Please provide path and filename to save figure. Else, set save_fig to false.')

    if axes:
        if save_fig and not fig:
            raise ValueError('Please provide path and filename to save figure. Else, set save_fig to false.')

        evoked_meg.plot(gfp=True, axes=axes, time_unit='s', spatial_colors=True, xlim=plot_xlim,
                        ylim=plot_ylim, show=display_figs)
        axes.vlines(x=0, ymin=axes.get_ylim()[0], ymax=axes.get_ylim()[1], color='grey', linestyles='--')

        if save_fig:
            save.fig(fig=fig, path=fig_path, fname=fname)

    else:
        fig = evoked_meg.plot(gfp=True, time_unit='s', spatial_colors=True, xlim=plot_xlim,
                              ylim=plot_ylim, show=display_figs)
        axes = fig.get_axes()[0]
        axes.vlines(x=0, ymin=axes.get_ylim()[0], ymax=axes.get_ylim()[1], color='grey', linestyles='--')

        if save_fig:
            save.fig(fig=fig, path=fig_path, fname=fname)


def evoked_topo(evoked_meg, picks, topo_times, title=None, fig=None, axes_ev=None, axes_topo=None, xlim=None, ylim=None,
                display_figs=False, save_fig=False, fig_path=None, fname=None):

    # Sanity check
    if save_fig and (not fig_path or not fname):
        raise ValueError('Please provide path and filename to save figure. Else, set save_fig to false.')

    if axes_ev and axes_topo:
        if save_fig and not fig:
            raise ValueError('Please provide path and filename to save figure. Else, set save_fig to false.')

        display_figs = True

        evoked_meg.plot_joint(times=topo_times, title=title, picks=picks, show=display_figs,
                              ts_args={'axes': axes_ev, 'xlim': xlim, 'ylim': ylim},
                              topomap_args={'axes': axes_topo})

        axes_ev.vlines(x=0, ymin=axes_ev.get_ylim()[0], ymax=axes_ev.get_ylim()[1], color='grey', linestyles='--')

        if save_fig:
            save.fig(fig=fig, path=fig_path, fname=fname)

    else:
        fig = evoked_meg.plot_joint(times=topo_times, title=title, picks=picks, show=display_figs,
                                    ts_args={'xlim': xlim, 'ylim': ylim})

        all_axes = plt.gcf().get_axes()
        axes_ev = all_axes[0]
        axes_ev.vlines(x=0, ymin=axes_ev.get_ylim()[0], ymax=axes_ev.get_ylim()[1], color='grey', linestyles='--')

        if save_fig:
            save.fig(fig=fig, path=fig_path, fname=fname)


def fig_vs_ms():

    fig = plt.figure(figsize=(10, 9))

    # 1st row Topoplots
    # VS
    ax1 = fig.add_axes([0.12, 0.88, 0.08, 0.09])
    ax2 = fig.add_axes([0.3, 0.88, 0.08, 0.09])
    ax3 = fig.add_axes([0.42, 0.88, 0.013, 0.09])

    #MS
    ax4 = fig.add_axes([0.7, 0.88, 0.08, 0.09])
    ax5 = fig.add_axes([0.82, 0.88, 0.013, 0.09])

    # 2nd row Evokeds
    ax6 = fig.add_axes([0.07, 0.71, 0.4, 0.15])
    ax7 = fig.add_axes([0.57, 0.71, 0.4, 0.15])

    # 3 row Topoplots
    # VS
    ax8 = fig.add_axes([0.2, 0.54, 0.08, 0.09])
    ax9 = fig.add_axes([0.32, 0.54, 0.013, 0.09])
    # MS
    ax10 = fig.add_axes([0.7, 0.54, 0.08, 0.09])
    ax11 = fig.add_axes([0.82, 0.54, 0.013, 0.09])

    # 4 row Evokeds
    ax12 = fig.add_axes([0.07, 0.38, 0.4, 0.15])
    ax13 = fig.add_axes([0.57, 0.38, 0.4, 0.15])

    # 5 row Topoplot Difference
    ax14 = fig.add_axes([0.4, 0.22, 0.08, 0.09])
    ax15 = fig.add_axes([0.52, 0.22, 0.013, 0.09])

    # 6th row Evoked Diference
    ax16 = fig.add_axes([0.1, 0.05, 0.8, 0.15])

    # groups
    ax_evoked_vs_1 = ax6
    ax_topo_vs_1 = [ax1, ax2, ax3]

    ax_evoked_ms_1 = ax7
    ax_topo_ms_1 = [ax4, ax5]

    ax_evoked_vs_2 = ax12
    ax_topo_vs_2 = [ax8, ax9]

    ax_evoked_ms_2 = ax13
    ax_topo_ms_2 = [ax10, ax11]

    ax_evoked_diff = ax16
    ax_topo_diff = [ax14, ax15]

    return fig, ax_evoked_vs_1, ax_topo_vs_1, ax_evoked_ms_1, ax_topo_ms_1, ax_evoked_vs_2, ax_topo_vs_2, \
           ax_evoked_ms_2, ax_topo_ms_2, ax_evoked_diff, ax_topo_diff


def fig_psd():

    fig = plt.figure(figsize=(15, 5))

    # 1st row Topoplots
    ax1 = fig.add_axes([0.05, 0.6, 0.15, 0.3])
    ax2 = fig.add_axes([0.225, 0.6, 0.15, 0.3])
    ax3 = fig.add_axes([0.4, 0.6, 0.15, 0.3])
    ax4 = fig.add_axes([0.575, 0.6, 0.15, 0.3])
    ax5 = fig.add_axes([0.75,  0.6, 0.15, 0.3])

    # 2nd row PSD
    ax6 = fig.add_axes([0.15,  0.1, 0.7, 0.4])

    # Group axes
    axs_topo = [ax1, ax2, ax3, ax4, ax5]
    ax_psd = ax6

    return fig, axs_topo, ax_psd


def fig_tf_bands(fontsize=None, ticksize=None):

    if fontsize:
        params = {'axes.titlesize': fontsize}
        plt.rcParams.update(params)
    if ticksize:
        params = {'axes.labelsize': ticksize, 'legend.fontsize': ticksize, 'xtick.labelsize': ticksize, 'ytick.labelsize': ticksize}
        plt.rcParams.update(params)

    fig, axes_topo = plt.subplots(3, 3, figsize=(15, 8), gridspec_kw={'width_ratios': [5, 1, 1]})

    for ax in axes_topo[:, 0]:
        ax.remove()
    axes_topo = [ax for ax_arr in axes_topo[:, 1:] for ax in ax_arr ]

    ax1 = fig.add_axes([0.1, 0.1, 0.5, 0.8])

    return fig, axes_topo, ax1


def tfr_bands(tfr, chs_id, plot_xlim=(None, None), baseline=None, bline_mode=None, dB=False, vmin=None, vmax=None, subject=None, title=None, vlines_times=[0],
              topo_times=None, display_figs=False, save_fig=False, fig_path=None, fname=None, fontsize=None, ticksize=None, cmap='bwr'):

    # Sanity check
    if save_fig and (not fname or not fig_path):
        raise ValueError('Please provide path and filename to save figure. Else, set save_fig to false.')

    if plot_xlim == (None, None):
        plot_xlim = (tfr.tmin, tfr.tmax)

    # Turn off dB if baseline mode is incompatible with taking log10
    if dB and bline_mode in ['mean', 'logratio']:
        dB = False

    # Define figure
    fig, axes_topo, ax_tf = fig_tf_bands(fontsize=fontsize, ticksize=ticksize)

    # Pick plot channels
    picks = functions_general.pick_chs(chs_id=chs_id, info=tfr.info)

    # Plot time-frequency
    tfr.plot(picks=picks, baseline=baseline, mode=bline_mode, tmin=plot_xlim[0], tmax=plot_xlim[1],
             combine='mean', cmap=cmap, axes=ax_tf, show=display_figs, vmin=vmin, vmax=vmax, dB=dB)

    # Plot time markers as vertical lines
    for t in vlines_times:
        try:
            ax_tf.vlines(x=t, ymin=ax_tf.get_ylim()[0], ymax=ax_tf.get_ylim()[1], linestyles='--', colors='gray')
        except:
            pass

    # Topomaps parameters
    if not topo_times:
        topo_times = plot_xlim
    topomap_kw = dict(ch_type='mag', tmin=topo_times[0], tmax=topo_times[1], baseline=baseline,
                      mode=bline_mode, show=display_figs)
    plot_dict = dict(Delta=dict(fmin=1, fmax=4), Theta=dict(fmin=4, fmax=8), Alpha=dict(fmin=8, fmax=12),
                     Beta=dict(fmin=12, fmax=30), Gamma=dict(fmin=30, fmax=45), HGamma=dict(fmin=45, fmax=100))

    # Plot topomaps
    for ax, (title_topo, fmin_fmax) in zip(axes_topo, plot_dict.items()):
        try:
            tfr.plot_topomap(**fmin_fmax, axes=ax, **topomap_kw)
        except:
            ax.text(0.5, 0.5, 'No data', horizontalalignment='center', verticalalignment='center')
            ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(title_topo)

    # Figure title
    if title:
        fig.suptitle(title)
    elif subject:
        fig.suptitle(subject.subject_id + f'_{fname.split("_")[0]}_{chs_id}_{bline_mode}_topotimes_{topo_times}')
    elif not subject:
        fig.suptitle(f'Grand_average_{fname.split("_")[0]}_{chs_id}_{bline_mode}_topotimes_{topo_times}')
        fname = 'GA_' + fname

    fig.tight_layout()

    if save_fig:
        fname += f'_topotimes_{topo_times}'
        os.makedirs(fig_path, exist_ok=True)
        save.fig(fig=fig, path=fig_path, fname=fname)


def fig_tf_times(time_len, figsize=(18, 5), ax_len_div=12, timefreqs_tfr=None, fontsize=None, ticksize=None):

    if fontsize:
        params = {'axes.titlesize': fontsize}
        plt.rcParams.update(params)
    if ticksize:
        params = {'axes.labelsize': ticksize, 'legend.fontsize': ticksize, 'xtick.labelsize': ticksize, 'ytick.labelsize': ticksize}
        plt.rcParams.update(params)


    fig = plt.figure(figsize=figsize)

    # # Define axes for topomaps
    if timefreqs_tfr:
        axes_topo = []
        for i in range(len(timefreqs_tfr)):
            width = 0.1 + 0.05
            start_pos = 0.075 + i * width
            ax = fig.add_axes([start_pos, 0.05, 0.1, 0.3])
            axes_topo.append(ax)

        # Define axis for colorbar
        start_pos = 0.075 + (i + 1) * width
        ax_topo_cbar = fig.add_axes([start_pos, 0.05, 0.005, 0.25])
    else:
        axes_topo = None
        ax_topo_cbar = None

    # Define TFR axis
    ax_tfr_len = time_len/ax_len_div
    ax_tfr = fig.add_axes([0.075, 0.55, ax_tfr_len, 0.3])
    ax_tfr_cbar = fig.add_axes([0.9, 0.05, 0.005, 0.3])

    return fig, axes_topo, ax_tfr, ax_topo_cbar, ax_tfr_cbar


def tfr_times(tfr, chs_id, timefreqs_tfr=None, plot_xlim=(None, None), baseline=None, bline_mode=None, dB=False, vmin=None, vmax=None,
              topo_vmin=None, topo_vmax=None, title=None, vlines_times=None, cmap='bwr', display_figs=False, save_fig=False, fig_path=None, fname=None, fontsize=None, ticksize=None):

    # Sanity check
    if save_fig and (not fname or not fig_path):
        raise ValueError('Please provide path and filename to save figure. Else, set save_fig to false.')

    if plot_xlim == (None, None):
        plot_xlim = (tfr.tmin, tfr.tmax)

    # Turn off dB if baseline mode is incompatible with taking log10
    if dB and bline_mode in ['mean', 'logratio']:
        dB = False

    # Apply baseline to tf plot data
    tfr_plot = tfr.copy().apply_baseline(baseline=baseline, mode=bline_mode)

    # Pick plot channels
    picks = functions_general.pick_chs(chs_id=chs_id, info=tfr.info)

    # Define colorbar limits if any missing
    if (vmax != None and vmin == None):
        vmin = -vmax
    elif (vmin != None and vmax == None):
        vmax = -vmin
    elif (vmin == None and vmax == None):
        vmin = tfr_plot.copy().pick(picks).data.mean(0).min()
        vmax = tfr_plot.copy().pick(picks).data.mean(0).max()
    
    if (topo_vmax != None and topo_vmin == None):
        topo_vmin = -topo_vmax
    elif (topo_vmin != None and topo_vmax == None):
        topo_vmax = -topo_vmin

    # Define figure
    time_len = plot_xlim[1] - plot_xlim[0]
    fig, axes_topo, ax_tf, ax_topo_cbar, ax_tfr_cbar = fig_tf_times(time_len=time_len, timefreqs_tfr=timefreqs_tfr, fontsize=fontsize, ticksize=ticksize)

    # Plot time-frequency
    tfr_plot.plot(picks=picks, tmin=plot_xlim[0], tmax=plot_xlim[1], combine='mean', cmap=cmap, axes=ax_tf, show=display_figs, vmin=vmin, vmax=vmax, dB=dB, colorbar=False)

    # Plot TF colorbar
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    # Get colorbar axis
    fig.colorbar(sm, cax=ax_tfr_cbar)
    ax_tfr_cbar.set_ylabel('Power (dB)')

    # Plot time markers as vertical lines
    if not vlines_times:
        vlines_times = [0]
    ymin = ax_tf.get_ylim()[0]
    ymax = ax_tf.get_ylim()[1]
    for t in vlines_times:
        try:
            ax_tf.vlines(x=t, ymin=ymin, ymax=ymax, linestyles='--', colors='gray')
        except:
            pass

    # Plot topomaps
    if axes_topo:
        # Get min and max from all topoplots
        if topo_vmin == None and topo_vmax == None:
            maxs = []
            if type(timefreqs_tfr) == dict:
                for topo_timefreqs in timefreqs_tfr.items():
                    tfr_crop = tfr_plot.copy().crop(tmin=topo_timefreqs[1]['tmin'], tmax=topo_timefreqs[1]['tmax'],
                                                    fmin=topo_timefreqs[1]['fmin'], fmax=topo_timefreqs[1]['fmax'])
                    data = tfr_crop.data.ravel()
                    maxs.append(data.max())
            elif type(timefreqs_tfr) == list:
                for topo_timefreqs in timefreqs_tfr:
                    tfr_crop = tfr_plot.copy().crop(tmin=topo_timefreqs[0], tmax=topo_timefreqs[0],
                                                    fmin=topo_timefreqs[1], fmax=topo_timefreqs[1])
                    data = tfr_crop.data.ravel()
                    maxs.append(data.max())
            topo_vmax = np.max(maxs)
            topo_vmin = -topo_vmax

        if type(timefreqs_tfr) == dict:
            for i, (ax, (key, topo_timefreqs)) in enumerate(zip(axes_topo, timefreqs_tfr.items())):
                # Topomaps parameters
                topomap_kw = dict(ch_type='mag', tmin=topo_timefreqs['tmin'], tmax=topo_timefreqs['tmax'],
                                  fmin=topo_timefreqs['fmin'], fmax=topo_timefreqs['fmax'], vlim=(topo_vmin, topo_vmax),
                                  cmap=cmap, colorbar=False, baseline=baseline,  mode=bline_mode, show=display_figs)

                try:
                    tfr.plot_topomap(axes=ax, **topomap_kw)
                except:
                    ax.text(0.5, 0.5, 'No data', horizontalalignment='center', verticalalignment='center')
                    ax.set_xticks([]), ax.set_yticks([])
                ax.set_title(topo_timefreqs['title'], color=f'C{i}', fontweight='bold')

        elif type(timefreqs_tfr) == list:
            for i, (ax, (topo_timefreqs)) in enumerate(zip(axes_topo, timefreqs_tfr)):
                # Topomaps parameters
                topomap_kw = dict(ch_type='mag', tmin=topo_timefreqs[0], tmax=topo_timefreqs[0],
                                  fmin=topo_timefreqs[1], fmax=topo_timefreqs[1], vlim=(topo_vmin, topo_vmax),
                                  cmap=cmap, colorbar=False, baseline=baseline, mode=bline_mode, show=display_figs)

                try:
                    tfr.plot_topomap(axes=ax, **topomap_kw)
                except:
                    ax.text(0.5, 0.5, 'No data', horizontalalignment='center', verticalalignment='center')
                    ax.set_xticks([]), ax.set_yticks([])
                ax.set_title(topo_timefreqs, color=f'C{i}', fontweight='bold')

        # Colorbar
        norm = matplotlib.colors.Normalize(vmin=topo_vmin, vmax=topo_vmax)
        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        # Get colorbar axis
        fig.colorbar(sm, cax=ax_topo_cbar)
        ax_topo_cbar.set_ylabel('Power (dB)')

    # Figure title
    if timefreqs_tfr:
        try:
            topo_times = [timefreq['tmin'] for (key, timefreq) in timefreqs_tfr.items()]
        except:
            topo_times = [timefreq[0] for timefreq in timefreqs_tfr]
    else:
        topo_times = None

    if title is None:
        title = fname + f'_topotimes_{topo_times}'
    fig.suptitle(title)

    if save_fig:
        fname += f'_topotimes_{topo_times}'
        os.makedirs(fig_path, exist_ok=True)
        save.fig(fig=fig, path=fig_path, fname=fname)

    return fig, ax_tf


def tfr_plotjoint(tfr, plot_baseline=None, bline_mode=None, plot_xlim=(None, None), timefreqs=None, plot_max=True, plot_min=True, vlines_times=None, cmap='bwr',
                  vmin=None, vmax=None, display_figs=False, save_fig=False, trf_fig_path=None, fname=None, fontsize=None, ticksize=None):
    # Sanity check
    if save_fig and (not fname or not trf_fig_path):
        raise ValueError('Please provide path and filename to save figure. Else, set save_fig to false.')

    if fontsize:
        params = {'axes.titlesize': fontsize}
        plt.rcParams.update(params)
    if ticksize:
        params = {'axes.labelsize': ticksize, 'legend.fontsize': ticksize, 'xtick.labelsize': ticksize, 'ytick.labelsize': ticksize}
        plt.rcParams.update(params)

    if plot_baseline:
        tfr_plotjoint = tfr.copy().apply_baseline(baseline=plot_baseline, mode=bline_mode)
    else:
        tfr_plotjoint = tfr.copy()

    # Get all mag channels to plot
    picks = functions_general.pick_chs(chs_id='mag', info=tfr.info)
    tfr_plotjoint = tfr_plotjoint.pick(picks)

    # Get maximum
    if not timefreqs:
        timefreqs = functions_analysis.get_plot_tf(tfr=tfr_plotjoint, plot_xlim=plot_xlim, plot_max=plot_max, plot_min=plot_min)

    # Title
    if fname:
        title = f'{fname.split("_")[1]}_{bline_mode}'
    else:
        title = f'{bline_mode}'

    fig = tfr_plotjoint.plot_joint(timefreqs=timefreqs, tmin=plot_xlim[0], tmax=plot_xlim[1], cmap=cmap, vmin=vmin, vmax=vmax,
                                   title=title, show=display_figs)

    # Plot vertical lines
    tf_ax = fig.axes[0]
    if not vlines_times:
        vlines_times = [0]
    for t in vlines_times:
        try:
            tf_ax.vlines(x=t, ymin=tf_ax.get_ylim()[0], ymax=tf_ax.get_ylim()[1], linestyles='--', colors='gray')
        except:
            pass

    if save_fig:
        save.fig(fig=fig, path=trf_fig_path, fname=fname)


def tfr_plotjoint_picks(tfr, plot_baseline=(None, 0), bline_mode=None, plot_xlim=(None, None), timefreqs=None, image_args=None, clusters_mask=None,
                        plot_max=True, plot_min=True, vmin=None, vmax=None, chs_id='mag', vlines_times=None, cmap='bwr',
                        display_figs=False, save_fig=False, trf_fig_path=None, fname=None, fontsize=None, ticksize=None):
    # Sanity check
    if save_fig and (not fname or not trf_fig_path):
        raise ValueError('Please provide path and filename to save figure. Else, set save_fig to false.')

    if fontsize:
        params = {'axes.titlesize': fontsize}
        plt.rcParams.update(params)
    if ticksize:
        params = {'axes.labelsize': ticksize, 'legend.fontsize': ticksize, 'xtick.labelsize': ticksize, 'ytick.labelsize': ticksize}
        plt.rcParams.update(params)

    if plot_baseline:
        tfr_plotjoint = tfr.copy().apply_baseline(baseline=plot_baseline, mode=bline_mode)
        tfr_topo = tfr.copy().apply_baseline(baseline=plot_baseline, mode=bline_mode)  # tfr for topoplots
    else:
        tfr_plotjoint = tfr.copy()
        tfr_topo = tfr.copy()  # tfr for topoplots

    # TFR from certain chs and topoplots from all channels
    picks = functions_general.pick_chs(chs_id=chs_id, info=tfr.info)
    tfr_plotjoint = tfr_plotjoint.pick(picks)

    # Get maximum
    if timefreqs == None:
        timefreqs = functions_analysis.get_plot_tf(tfr=tfr_plotjoint, plot_xlim=plot_xlim, plot_max=plot_max, plot_min=plot_min)

    # Title
    if fname:
        title = f'{fname.split("_")[1]}_{bline_mode}'
    else:
        title = f'{bline_mode}'

    # Get min and max from all topoplots and use in TF plot aswell
    if vmin == None or vmax == None:
        maxs = []
        mins = []
        for timefreq in timefreqs:
            tfr_crop = tfr_topo.copy().crop(tmin=timefreq[0], tmax=timefreq[0], fmin=timefreq[1], fmax=timefreq[1])
            data = tfr_crop.data.ravel()
            mins.append(- 2 * data.std())
            maxs.append(2 * data.std())
        vmax = np.max(maxs)
        vmin = np.min(mins)

    # Plot tf plot joint
    fig = tfr_plotjoint.plot_joint(timefreqs=timefreqs, tmin=plot_xlim[0], tmax=plot_xlim[1], cmap=cmap, image_args=image_args,
                                   title=title, show=display_figs, vmin=vmin, vmax=vmax)

    # Plot vertical lines
    tf_ax = fig.axes[0]
    if not vlines_times and (tfr_plotjoint.times[0] != 0 and tfr_plotjoint.times[-1] != 0):
        vlines_times = [0]
    for t in vlines_times:
        try:
            tf_ax.vlines(x=t, ymin=tf_ax.get_ylim()[0], ymax=tf_ax.get_ylim()[1], linestyles='--', colors='gray')
        except:
            pass

    # Define Topo mask
    if clusters_mask is not None:
        # Define significant channels to mask in time freq interval around desired tf point
        topo_mask = [clusters_mask[functions_general.find_nearest(tfr.freqs, timefreq[1] - 3)[0]:
                                   functions_general.find_nearest(tfr.freqs, timefreq[1] + 1)[0],
                     functions_general.find_nearest(tfr.copy().crop(tmin=plot_xlim[0], tmax=plot_xlim[1]).times, timefreq[0] - 0.05)[0]:
                     functions_general.find_nearest(tfr.copy().crop(tmin=plot_xlim[0], tmax=plot_xlim[1]).times, timefreq[0] + 0.05)[0]].
                     sum(axis=0).sum(axis=0).astype(bool) for timefreq in timefreqs]
        masks = []
        for topo in topo_mask:
            mask = np.zeros(len(tfr_topo.copy().pick('mag').info.ch_names)).astype(bool)
            mask[[idx for idx, channel in enumerate(tfr_topo.copy().pick('mag').info.ch_names) if channel in list(compress(tfr_plotjoint.info.ch_names, topo))]] = True
            masks.append(mask)
    else:
        masks = [None]*len(timefreqs)
    mask_params = dict(marker='o', markerfacecolor='white', markeredgecolor='k', linewidth=0, markersize=4, alpha=0.6)

    # Get topo axes and overwrite topoplots
    topo_axes = fig.axes[1:-1]
    for i, (ax, timefreq) in enumerate(zip(topo_axes, timefreqs)):
        ax.clear()
        topomap_kw = dict(ch_type='mag', tmin=timefreq[0], tmax=timefreq[0], fmin=timefreq[1], fmax=timefreq[1], mask=masks[i], mask_params=mask_params, colorbar=False, show=display_figs)
        tfr_topo.plot_topomap(axes=ax, cmap=cmap, vlim=(vmin, vmax), **topomap_kw)

    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    # Get colorbar axis
    cbar_ax = fig.axes[-1]
    fig.colorbar(sm, cax=cbar_ax)

    if save_fig:
        save.fig(fig=fig, path=trf_fig_path, fname=fname)


def mri_meg_alignment(subject, subject_code, dig, subjects_dir=os.path.join(paths.mri_path, 'FreeSurfer_out')):

    # Path to MRI <-> HEAD Transformation (Saved from coreg)
    trans_path = os.path.join(subjects_dir, subject_code, 'bem', f'{subject_code}-trans.fif')
    fids_path = os.path.join(subjects_dir, subject_code, 'bem', f'{subject_code}-fiducials.fif')
    dig_info_path = paths.opt_path + subject.subject_id + '/info_raw.fif'

    # Load raw meg data with dig info
    info_raw = mne.io.read_raw_fif(dig_info_path)

    # Visualize MEG/MRI alignment
    surfaces = dict(brain=0.7, outer_skull=0.5, head=0.4)
    # Try plotting with head skin and brain
    try:
        mne.viz.plot_alignment(info_raw.info, trans=trans_path, subject=subject_code,
                                     subjects_dir=subjects_dir, surfaces=surfaces,
                                     show_axes=True, dig=dig, eeg=[], meg='sensors',
                                     coord_frame='meg', mri_fiducials=fids_path)
    # Plot only outer skin
    except:
        mne.viz.plot_alignment(info_raw.info, trans=trans_path, subject=subject_code,
                                     subjects_dir=subjects_dir, surfaces='outer_skin',
                                     show_axes=True, dig=dig, eeg=[], meg='sensors',
                                     coord_frame='meg', mri_fiducials=fids_path)


def global_mean_con(subject, mean_global_con, subject_code=None, save_fig=False, fig_path=None, fname=None):
    # Sanity check
    if save_fig and (not fig_path):
        raise ValueError('Please provide path and filename to save figure. Else, set save_fig to false.')

    means = [np.mean(mean_global_con[key]) for key in mean_global_con]
    std_devs = [np.std(mean_global_con[key], ddof=1) for key in mean_global_con.keys()]  # Sample std deviation

    # Create bar plot
    fig = plt.figure(figsize=(6, 4))
    plt.bar(mean_global_con.keys(), means, yerr=std_devs, capsize=5, color=['blue', 'blue'], alpha=0.7)

    # Labels and title
    plt.ylabel("Mean Value")
    plt.title("Mean and Standard Deviation of DA and CF")

    # Save
    if save_fig:
        if not fname:
            fname = f'{subject.subject_id}_circle'
        if subject_code == 'fsaverage':
            fname += '_fsaverage'
        save.fig(fig=fig, path=fig_path, fname=fname)


def connectivity_circle(subject, labels, con, surf_vol, vmin=None, vmax=None, connectivity_method='pli', n_lines=100, subject_code=None, display_figs=False, save_fig=False,
                        fig_path=None, fname=None, fontsize=None, ticksize=None):
    # Sanity check
    if save_fig and (not fig_path):
        raise ValueError('Please provide path and filename to save figure. Else, set save_fig to false.')

    # Plot fonsize params
    if fontsize:
        params = {'axes.titlesize': fontsize}
        plt.rcParams.update(params)
    if ticksize:
        params = {'axes.labelsize': ticksize, 'legend.fontsize': ticksize, 'xtick.labelsize': ticksize, 'ytick.labelsize': ticksize}
        plt.rcParams.update(params)

    # Get colors for each label
    label_colors = [label.color for label in labels]

    # Reorder the labels based on their location in the left hemi
    label_names = [label.name for label in labels]
    lh_labels = [name for name in label_names if ('lh' in name or 'Left' in name)]

    # Get the y-location of the label
    label_ypos = list()
    for name in lh_labels:
        idx = label_names.index(name)
        ypos = np.mean(labels[idx].pos[:, 1])
        label_ypos.append(ypos)

    # Reorder the labels based on their location
    lh_labels = [label for (yp, label) in sorted(zip(label_ypos, lh_labels))]

    # For the right hemi
    rh_labels = [label.replace('-lh', '-rh') for label in lh_labels]

    # Save the plot order and create a circular layout
    node_order = list()
    # include first volume nodes that are in neither hemisphere
    if surf_vol == 'volume':
        node_order.extend([label for label in label_names if label not in lh_labels and label not in rh_labels])
    node_order.extend(lh_labels[::-1])  # reverse the order
    node_order.extend(rh_labels)

    node_angles = mne.viz.circular_layout(label_names, node_order, start_pos=90, group_boundaries=[0, len(label_names) / 2])

    # Plot the graph using node colors from the FreeSurfer parcellation
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='black', subplot_kw=dict(polar=True))

    # Plot
    if not vmin:
        sorted_con = con.flatten()
        sorted_con.sort()
        if isinstance(n_lines, int):
            vmin = sorted_con[-n_lines]
        elif isinstance(n_lines, float):
            vmin = sorted_con[-int(n_lines*len(label_names))]

    if not vmax:
        sorted_con = con.flatten()
        sorted_con.sort()
        vmax = sorted_con[-1]

    mne_connectivity.viz.plot_connectivity_circle(con, label_names, n_lines=n_lines, vmin=vmin, vmax=vmax,
                                                  node_angles=node_angles, node_colors=label_colors,
                                                  title=f'All-to-All Connectivity ({connectivity_method})', ax=ax,
                                                  show=display_figs)
    fig.tight_layout()

    # Save
    if save_fig:
        if not fname:
            fname = f'{subject.subject_id}_circle'
        if subject_code == 'fsaverage':
            fname += '_fsaverage'
        save.fig(fig=fig, path=fig_path, fname=fname)


def connectome(subject, labels, adjacency_matrix, subject_code, save_fig=False, fig_path=None, fname='connectome', connections_num=150,
               template='MNI152NLin2009cAsym', template_style='glass', template_alpha=10, node_alpha=0.5, node_scale=30, node_color='black',
               edge_alpha=0.7, edge_colorvminvmax='absmax', edge_thresholddirection='above', edge_cmap='coolwarm', edge_widthscale=0.7, view='preset-3'):

    # Sanity check
    if save_fig and (not fig_path):
        raise ValueError('Please provide path and filename to save figure. Else, set save_fig to false.')

    # Get parcelation labels positions
    label_names = [label.name for label in labels]
    label_xpos = list()
    label_ypos = list()
    label_zpos = list()
    for name in [label.name for label in labels]:
        idx = label_names.index(name)
        label_xpos.append(np.mean(labels[idx].pos[:, 0]) * 1100)
        label_ypos.append(np.mean(labels[idx].pos[:, 1]) * 1100)
        label_zpos.append(np.mean(labels[idx].pos[:, 2]) * 1100)

    nodes_df = pd.DataFrame(data={'x': label_xpos,
                                  'y': label_ypos,
                                  'z': label_zpos})

    # Plot only if at least 1 connection
    if abs(adjacency_matrix).sum() > 0:
        # Define threshold based on connections number
        # Get the values that exceed the thresh and define edges to plot
        # Extract retained edge values for vmin/vmax
        if connections_num > 1:
            if edge_thresholddirection == 'absabove':
                edge_threshold = sorted(np.sort(np.abs(adjacency_matrix), axis=None))[-int(connections_num * 2) - 1]
                retained_edges = adjacency_matrix[np.abs(adjacency_matrix) > edge_threshold]
            elif edge_thresholddirection == 'above':
                edge_threshold = sorted(np.sort(adjacency_matrix, axis=None))[-int(connections_num * 2) - 1]
                retained_edges = adjacency_matrix[adjacency_matrix > edge_threshold]
            elif edge_thresholddirection == 'below':
                edge_threshold = sorted(np.sort(adjacency_matrix, axis=None))[int(connections_num * 2) + 1]
                retained_edges = adjacency_matrix[adjacency_matrix < edge_threshold]
        else:
            edge_threshold = connections_num
            if edge_thresholddirection == 'absabove':
                retained_edges = adjacency_matrix[np.abs(adjacency_matrix) > edge_threshold]
            elif edge_thresholddirection == 'above':
                retained_edges = adjacency_matrix[adjacency_matrix > edge_threshold]
            elif edge_thresholddirection == 'below':
                retained_edges = adjacency_matrix[adjacency_matrix < edge_threshold]

        # Determine vmin and vmax from retained edges
        # Option 1: Signed values (for diverging colormap)
        if edge_colorvminvmax == 'absmax':
            vmax = np.max(abs(retained_edges))
            vmin = - vmax
        elif edge_colorvminvmax == 'minmax':
            vmin = np.min(retained_edges)
            vmax = np.max(retained_edges)
        else:
            raise NameError(f"edge_colorvminvmax should be either 'absmax' or 'minmax'")

        if edge_cmap is None:
            if vmin < 0:
                edge_cmap = 'coolwarm'
            else:
                edge_cmap = 'YlOrRd'

        # Corresponding node labels
        nodes = [i for i in range(len(adjacency_matrix))]

        # Convert matrix to long format (excluding diagonal)
        i, j = np.triu_indices_from(adjacency_matrix, k=1)  # Get upper triangular indices
        weights = adjacency_matrix[i, j]  # Get connectivity values

        # Create DataFrame
        weights_col_name = "weight"
        connectivity_df = pd.DataFrame({
            "i": [nodes[idx] for idx in i],  # Convert indices to node names
            "j": [nodes[idx] for idx in j],
            weights_col_name: weights
        })

        # Call netplotbrain to plot
        fig, ax = netplotbrain.plot(
            template=template,
            template_style=template_style,
            template_alpha=template_alpha,
            nodes=nodes_df,
            node_alpha=node_alpha,
            node_scale=node_scale,
            node_color=node_color,
            edges=adjacency_matrix,
            edges_df=connectivity_df,
            edge_alpha=edge_alpha,
            edge_thresholddirection=edge_thresholddirection,
            edge_threshold=edge_threshold,
            edge_color=weights_col_name,
            edge_cmap=edge_cmap,
            edge_colorvminvmax=[vmin, vmax],
            edge_widthscale=edge_widthscale,
            view=view,
        )

        # Add a colorbar with calculated vmin and vmax
        sm = plt.cm.ScalarMappable(cmap=edge_cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax[-1], label='Connectivity Strength', shrink=0.5, aspect=20)

        # Show the plot
        plt.tight_layout()

        # Save
        if save_fig:
            if not fname:
                fname = f'{subject.subject_id}_connectome'
            if subject_code == 'fsaverage' and 'fsaverage' not in fname:
                fname += '_fsaverage'
            save.fig(fig=fig, path=fig_path, fname=fname)


def plot_con_matrix(subject, labels, adjacency_matrix, subject_code, save_fig=False, fig_path=None, fname='matrix'):

    # Sanity check
    if save_fig and (not fig_path):
        raise ValueError('Please provide path and filename to save figure. Else, set save_fig to false.')

    # Get labels names
    label_names = [label.name for label in labels]

    # Get the y-location of the label
    label_ypos = np.array([np.mean(labels[label_names.index(name)].pos[:, 1]) for name in label_names])

    # Separate left and right hemisphere labels
    left_indices = [i for i, name in enumerate(label_names) if 'lh' in name]
    right_indices = [i for i, name in enumerate(label_names) if 'rh' in name]

    # Sort within each hemisphere (anterior-to-posterior)
    left_sorted = sorted(left_indices, key=lambda i: label_ypos[i], reverse=True)  # Anterior to posterior
    right_sorted = sorted(right_indices, key=lambda i: label_ypos[i], reverse=True)  # Anterior to posterior

    # Combine order: left first, then right
    sorted_indices = left_sorted + right_sorted

    # Reorder the connectivity matrix and labels
    sorted_matrix = adjacency_matrix[sorted_indices, :][:, sorted_indices]
    sorted_labels = [label_names[i] for i in sorted_indices]  # Get labels in sorted order

    # # Make adjacency matrix sorted from frontal to posterior
    # sort = np.argsort(label_ypos)  # Get sorted indexes based on regions anterior-posterior order
    # sorted_matrix = adjacency_matrix[sort[::-1]]  # Sort on one axis
    # sorted_matrix = sorted_matrix[:, sort[::-1]]  # Apply same sort on second axis
    #
    # sorted_labels = [label_names[i] for i in sort][::-1]  # Get labels in sorted order

    # Get min and max from data
    vmin = np.sort(sorted_matrix.ravel())[len(label_ypos)]
    vmax = np.sort(sorted_matrix.ravel())[-1]

    # Plot
    fig = plt.figure(figsize=(8, 5))
    norm = colors.CenteredNorm(vcenter=0)
    im = plt.imshow(sorted_matrix, cmap='coolwarm', norm=norm)
    # im = plt.imshow(sorted_matrix, vmin=vmin, vmax=vmax)
    fig.colorbar(im)

    ax = plt.gca()
    ax.set_yticklabels(sorted_labels)
    ax.set_xticklabels([])

    plt.suptitle('')

    # Save
    if save_fig:
        if not fname:
            fname = f'{subject.subject_id}_matrix'
        if subject_code == 'fsaverage' and 'fsaverage' not in fname:
            fname += '_fsaverage'
        save.fig(fig=fig, path=fig_path, fname=fname)

    return sorted_matrix


def connectivity_strength(subject, subject_code, con, src, labels, surf_vol, subjects_dir, labels_fname, label_names_segmentation,
                          save_fig, fig_path, fname, threshold=1):

    try:
        mne.viz.close_all_3d_figures()
    except:
        pass

    # Define save file name
    if not fname:
        fname = f'{subject.subject_id}_strength'
    if subject_code == 'fsaverage' and 'fsaverage' not in fname:
        fname += '_fsaverage'

    # Plot connectivity strength (connections from each region to other regions)
    degree = mne_connectivity.degree(connectivity=con, threshold_prop=threshold)
    if surf_vol == 'surface':
        stc = mne.labels_to_stc(labels, degree, src=src)
        stc = stc.in_label(mne.Label(src[0]["vertno"], hemi="lh") + mne.Label(src[1]["vertno"], hemi="rh"))
        hemi = 'split'
        views = ['lateral', 'med']
        brain = stc.plot(src=src, subject=subject_code, subjects_dir=subjects_dir, size=(1000, 500), clim=dict(kind="percent", lims=[50, 75, 95]), hemi=hemi, views=views)

    if surf_vol == 'volume':
        # stc = mne.VolSourceEstimate(degree, [src[0]["vertno"]], 0, 1, "bst_resting")
        stc = mne.labels_to_stc(labels=(labels_fname, label_names_segmentation), values=degree, src=src)
        # stc = stc.in_label(mne.Label(src[0]["vertno"]), mri=labels_fname, src=src)
        hemi = 'both'
        views = 'dorsal'
        try:
            brain = stc.plot_3d(src=src, subject=subject_code, subjects_dir=subjects_dir, size=(1000, 500), clim=dict(kind="percent", lims=[40, 75, 95]), hemi=hemi, views=views)
        except:
            brain = stc.plot_3d(src=src, subject=subject_code, subjects_dir=subjects_dir, size=(1000, 500), hemi=hemi, views=views)

    if save_fig:
        brain.save_image(filename=fig_path + fname + '.png')
        brain.save_image(filename=fig_path + '/svg/' + fname + '.pdf')


def sources(stc, src, subject, subjects_dir, initial_time, surf_vol, force_fsaverage, source_estimation, save_fig, fig_path, fname, surface='pial', hemi='split', views='lateral',
            alpha=0.75, mask_negatives=False, time_label='auto', save_vid=True, positive_cbar=None, clim=None):

    # Close all plot figures
    try:
        mne.viz.close_all_3d_figures()
        plt.close('all')
    except:
        pass

    # Convert view to list in case only 1 view as str
    if isinstance(views, str):
        views = [views]

    # Get time indexes matching selected time
    if initial_time != None:
        initial_time_idx, _ = functions_general.find_nearest(array=stc.times, values=initial_time)
    else:
        initial_time_idx = None

    if source_estimation != 'cov':
        # If times to plot is list, average data in time interval
        if isinstance(initial_time, list) and len(initial_time) == 2:
            time_data = stc.data[:, initial_time_idx[0]:initial_time_idx[1]]
            average_time_data = time_data.mean(axis=1)
            # Overwrite data in first initial time
            stc.data[:, initial_time_idx[0]] = average_time_data
            stc.data[:, initial_time_idx[1]] = average_time_data
            # Set initial time to plot first initial time
            initial_time_plot = initial_time[0]

        # if float just plot at selected time
        else:
            initial_time_plot = initial_time
    # Set plot times as None for cov method
    else:
        initial_time_plot = None

    # Define clim
    if not clim:

        clim = {'kind': 'values', 'lims': ((abs(stc.data[:, initial_time_idx]).max() - abs(stc.data[:, initial_time_idx]).min()) / 1.5,
                                           (abs(stc.data[:, initial_time_idx]).max() - abs(stc.data[:, initial_time_idx]).min()) / 1.25,
                                           (abs(stc.data[:, initial_time_idx]).max() - abs(stc.data[:, initial_time_idx]).min()) * 1.1)}

        # Replace positive cbar for positive / negative
        if positive_cbar == False or (stc.data[:, initial_time_idx].mean() - stc.data[:, initial_time_idx].std() <= 0 and positive_cbar != True):
            clim['pos_lims'] = clim.pop('lims')


    if surf_vol == 'volume':

        # Nutmeg plot
        if subject == 'fsaverage':
            bg_img = paths.mri_path + 'MNI_templates/tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz'
        else:
            bg_img = None

        # Set backend
        matplotlib.use('TkAgg')

        fig = stc.plot(src=src, subject=subject, subjects_dir=subjects_dir, initial_time=initial_time_plot, clim=clim, bg_img=bg_img)
        if save_fig:
            if force_fsaverage:
                fname += '_fsaverage'
            if mask_negatives:
                fname += '_masked'
            if initial_time != None and not source_estimation == 'cov':
                fname += f'_{initial_time}'
            os.makedirs(fig_path, exist_ok=True)
            save.fig(fig=fig, path=fig_path, fname=fname)

        # 3D plot
        brain = stc.plot_3d(src=src, subject=subject, subjects_dir=subjects_dir, hemi=hemi, views=views, clim=clim,
                            initial_time=initial_time_plot, size=(1000, 500), time_label=time_label, brain_kwargs=dict(surf=surface, alpha=alpha))

        if save_fig:
            view_fname = fname + f'_3D'
            if force_fsaverage:
                view_fname += '_fsaverage'
            if mask_negatives:
                view_fname += '_masked'
            os.makedirs(fig_path + '/svg/', exist_ok=True)
            brain.save_image(filename=fig_path + view_fname + '.png')
            brain.save_image(filename=fig_path + '/svg/' + view_fname + '.pdf')
            if save_vid and not source_estimation == 'cov':
                try:
                    brain.save_movie(filename=fig_path + view_fname + '.mp4', time_dilation=12, framerate=30)
                except:
                    pass

    # 3D plot
    elif surf_vol == 'surface':

        brain = stc.plot(src=src, subject=subject, subjects_dir=subjects_dir, hemi=hemi, clim=clim, initial_time=initial_time_plot, views=views, size=(1000, 500),
                         brain_kwargs=dict(surf=surface, alpha=alpha))

        if save_fig:
            view_fname = fname + f'_3D'
            if force_fsaverage:
                view_fname += '_fsaverage'
            if mask_negatives:
                view_fname += '_masked'
            os.makedirs(fig_path + '/svg/', exist_ok=True)
            brain.save_image(filename=fig_path + view_fname + '.png')
            brain.save_image(filename=fig_path + '/svg/' + view_fname + '.pdf')
            if save_vid and not source_estimation == 'cov':
                try:
                    brain.save_movie(filename=fig_path + view_fname + '.mp4', time_dilation=12, framerate=30)
                except:
                    pass

    return brain


def source_tf(tf, clusters_mask_plot=None, vlim=(None, None), cmap='RdBu_r', hist_data=None, display_figs=False, save_fig=True, fig_path=None, fname=None, title=None):

    # Sanity check
    if save_fig and (not fig_path):
        raise ValueError('Please provide path and filename to save figure. Else, set save_fig to false.')

    if isinstance(hist_data, pd.Series):
        # Create the figure and gridspec for custom subplot sizes
        fig = plt.figure(figsize=(10, 4))

        # Define gridspec
        gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 0.1])

        # Upper subplot
        ax1 = fig.add_subplot(gs[0, 0])  # First row, first column
        tf.plot(axes=ax1, combine='mean', cmap=cmap, vmin=vlim[0], vmax=vlim[1], mask=clusters_mask_plot, mask_style='contour', show=display_figs, title=title)[0]
        ax1.vlines(x=0, ymin=ax1.get_ylim()[0], ymax=ax1.get_ylim()[1], linestyles='--', colors='gray')

        # Capture the position of the modified axis
        pos1 = ax1.get_position()

        # Calculate the position for the second subplot
        # Use the same width and align it with the adjusted position of the first subplot
        ax2_bottom = pos1.y0 - (pos1.height / 2)  # Place below ax1, taking 1/3 of the height
        ax2_height = pos1.height / 3

        # Add the lower subplot using the captured size
        ax2 = fig.add_axes([pos1.x0, ax2_bottom, pos1.width, ax2_height], sharex=ax1)
        ax2.hist(hist_data, range=(tf.tmin, tf.tmax), bins=50, edgecolor='black', linewidth=0.3, stacked=True)
        ax2.set_xlabel('Time (s)')

        # Remove the x-axis labels of the first plot
        ax1.tick_params(labelbottom=False)
        ax1.set_xlabel("")  # Remove the x-axis label

    else:
        fig, ax = plt.subplots(figsize=(10, 7))
        fig = tf.plot(axes=ax, cmap=cmap, vmin=vlim[0], vmax=vlim[1], mask=clusters_mask_plot, mask_style='contour', show=display_figs, combine='mean', title=title)[0]
        ax.vlines(x=0, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], linestyles='--', colors='gray')

    fig.suptitle(fname)

    if save_fig:
        save.fig(fig=fig, path=fig_path, fname=fname)
        if isinstance(clusters_mask_plot, np.ndarray):
            if 'sig/' not in fig_path:
                save.fig(fig=fig, path=fig_path + 'sig/', fname=fname)
            else:
                save.fig(fig=fig, path=fig_path, fname=fname)
    return fig


def average_tf_and_significance_heatmap(generic_tfr, sig_tfr, sig_mask, sig_regions, sig_chs_percent, hist_data, active_times, l_freq, h_freq, display_figs, save_fig, fig_path):

    # Quadrant average
    avg_tfr = generic_tfr.copy()
    avg_tfr.data = np.array([tfr.data for tfr in sig_tfr]).mean(axis=0)

    # Quadrant clusters
    min_sig_chs = sig_chs_percent * len(sig_regions)
    mask = np.array(sig_mask).sum(axis=0) > min_sig_chs

    # Define figure name
    fname = f'Avg'
    title = fname
    if active_times:
        fname += f"_{active_times[0]}_{active_times[1]}"

    # Plot
    fig = source_tf(tf=avg_tfr, clusters_mask_plot=mask, hist_data=hist_data, display_figs=display_figs,
                                 save_fig=save_fig, fig_path=fig_path, fname=fname, title=title)

    # Make number of signifficant regions heatmap
    sig_mask_array = np.expand_dims(np.array(sig_mask).sum(axis=0), axis=0)

    # Significance heatmap
    sig_heatmap = generic_tfr.copy()
    sig_heatmap.data = sig_mask_array

    # Define figure name
    fname = f'Sig_regions_{l_freq}_{h_freq}'
    title = fname
    if active_times:
        fname += f"_{active_times[0]}_{active_times[1]}"

    # Plot
    fig = source_tf(tf=sig_heatmap, vlim=(0, max(sig_mask_array.ravel())), cmap='Reds',
                                 clusters_mask_plot=mask, hist_data=hist_data, display_figs=display_figs,
                                 save_fig=save_fig, fig_path=fig_path, fname=fname, title=title)

    # Close figure
    if not display_figs:
        plt.close('all')


def add_task_lines(y_text, fontsize=10, color='white', ax=None):
    lines_times = [0, 2, 3]
    text_times = [-.32, 1, 2.5, 3.75]
    x_conditions = ['Cross 1', 'MS', 'Cross 2', 'VS']
    if ax is None: ax = plt.gca()  # noqa
    plt.sca(ax)
    for line_time in lines_times:
        plt.axvline(line_time, lw=2, color=color)
    for x_t, t_t in zip(text_times, x_conditions):
        plt.text(x_t, y_text, t_t, color=color, fontsize=fontsize, ha='center',
                 va='center', fontweight='bold')