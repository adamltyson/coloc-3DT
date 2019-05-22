"""
Adam Tyson | adamltyson@gmail.com | 2019-05-22

"""


def opt_initialise_all():
    opt_names = [
        'bleach_correct',
        'normalise',
        'enrichment',
        'seg_plot',
        'final_plot',
        'reslice',
        'save_results',
        'save_vars',
        'save_seg',
        'test'
    ]

    opt_prompts = [
        'Bleach correction?',
        'Normalise colocalisation to baseline (t=0)?',
        'Measure enrichment?',
        'Segmentation checker?',
        'Plot final results?',
        'Reslice to isotropic resolution?',
        'Save results as .csv?',
        'Save analysis options as .txt?',
        'Save segmentation as tif?',
        'Testing?'
    ]

    opt_defaults = [
        False,
        False,
        True,
        False,
        False,
        True,
        False,
        True,
        False,
        False
    ]

    return opt_names, opt_prompts, opt_defaults


def var_initialise_all():
    var_names = [
        'marker_smooth',
        'marker_thresh',
        'marker_noise_rem',
        'marker_dilate_margin',
        'marker_dilate_dist',
        'reslice_order',
        'plot_size',
        'num_coloc_channels',
        'max_frames'
    ]

    var_prompts = [
        'Marker smoothing width (pixels)',
        'Marker threshold adjustment (a.u)',
        'Marker noise removal (pixels)',
        'Marker normalisation mask (non-marker) erosion (pixels)',
        'Cell size estimation radius (pixels)',
        'Reslicing interpolation order',
        'Plot size (a.u.)',
        'Number colocalisation channels',
        'Number of frames to analyse (0 for all)'
                    ]

    var_defaults = [
        1,
        1.08,
        1,
        1,
        15,
        1,
        3,
        1,
        0,
    ]

    return var_names, var_prompts, var_defaults


def var_force(variable_dict):
    if 'marker_noise_rem' in variable_dict.keys():
        variable_dict['marker_noise_rem'] =\
            int(round(variable_dict['marker_noise_rem']))

    if 'marker_dilate_margin' in variable_dict.keys():
        variable_dict['marker_dilate_margin'] =\
            int(round(variable_dict['marker_dilate_margin']))

    if 'marker_dilate_dist' in variable_dict.keys():
        variable_dict['marker_dilate_dist'] =\
            int(round(variable_dict['marker_dilate_dist']))

    if 'reslice_order' in variable_dict.keys():
        variable_dict['reslice_order'] =\
            int(round(variable_dict['reslice_order']))
        if variable_dict['reslice_order'] < 1:
            variable_dict['reslice_order'] = 1

    if 'max_frames' in variable_dict.keys():
        variable_dict['max_frames'] =\
            int(round(variable_dict['max_frames']))

    if 'marker_med_filt_rad' in variable_dict.keys():
        variable_dict['marker_med_filt_rad'] =\
            int(round(variable_dict['marker_med_filt_rad']))

    if 'num_coloc_channels' in variable_dict.keys():
        variable_dict['num_coloc_channels'] =\
            int(round(variable_dict['num_coloc_channels']))
        if variable_dict['num_coloc_channels'] < 1:
            variable_dict['num_coloc_channels'] = 1

    if variable_dict['plot_size'] < 2:
        variable_dict['plot_size'] = 2

    return variable_dict


def channel_initialise(num_coloc_channels=1):
    channel_names = ['struct_channel']
    channel_prompts = ['Structural marker channel']
    channel_defaults = ['T00_C2.tif']

    for coloc_channel in range(0, num_coloc_channels):
        channel_names.append('coloc_channel_' + str(coloc_channel))
        channel_prompts.append(
            'Colocalisation marker channel ' + str(coloc_channel))
        channel_defaults.append('T00_C0.tif')

    return channel_names, channel_prompts, channel_defaults
