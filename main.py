"""
Adam Tyson | adamltyson@gmail.com | 2019-05-22

Takes a series of image stacks, exported from Slidebook
One 3D image per timepoint (T) and channel (C) (*Txx_Cxx.tif)
struct_channel = Marker, e.g. biological structure
coloc_channel_n = N m measurement channels, e.g. protein levels
Segments marker and calculates coloc_channel levels in marker over time

OPTIONS:
        'Bleach correction?' (Blind correction for segmentation, then corrects
                              colocalisation levels for mean in-cell intensity)
        'Normalise colocalisation to baseline (t=0)?'
        'Measure enrichment?'
        'Segmentation checker?' (Scroll plots for marker segmentation)
        'Plot final results?'
        'Reslice to isotropic resolution?'
        'Save results as .csv?'
        'Save analysis options as .txt?'
        'Save segmentation as tif?'
        'Testing?' (Variable, for development)

VARIABLES:
        'Marker smoothing width (pixels)',
        'Marker threshold adjustment (a.u)',
        'Marker noise removal (pixels)',
        'Marker normalisation mask (non-marker) erosion (pixels)',
        'Cell size estimation radius (pixels)'
        'Reslicing interpolation order',
        'Plot size (a.u.)',
        'Number colocalisation channels'
        'Number of frames to analyse (0 for all)'

OUTPUTS:
    *channel*_colocalisation.csv         - mean level of channel in the marker
                                 (adjusted wrt above parameters)
    *channel*_total_in_marker.csv       - total level of channel in marker
                                    (like colocalisation,
                                    (but the sum, and not mean)
    sum_marker_um3.csv            - total marker vol (in real units)
    sum_marker_voxels.csv         - total marker vol (in voxels, which
                                 may have been resliced)


REQUIRES:
    matplotlib
    scikit-image
    pillow
    numpy
    tk
    scipy
    pandas
    seaborn

IF ERROR: "Process finished with exit code -1073741819 (0xC0000005)"
    - linked to tkinter, matplotlib backends, pycharm, and maybe non-mirrored
      multiple monitors
    - to fix, make a fresh conda environment
    - needs TkAgg backend for tk, then Qt5Agg backend for plotting

"""

from datetime import datetime
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import coloc.GUI as GUI
import coloc.tools as tools
import coloc.analysis as analysis
import glob
import logging


def non_gui_set(opt, var):
    options['print_logging_level'] = 'INFO'
    options['file_logging_level'] = 'DEBUG'

    variables['save_attrs'] = ['*colocalisation*',
                               '*total*',
                               '*sum*']
    variables['save_heading'] = 'filename'
    variables['summary_plot_attrs'] = ['*colocalisation*']
    return opt, var


if __name__ == '__main__':
    options, variables, direc, channel_info = GUI.gui_run()
    options, variables = non_gui_set(options, variables)

    tools.setup_logging('coloc-3DT', print_level=options['print_logging_level'],
                        file_level=options['file_logging_level'])
    start_time = datetime.now()

    filenames = sorted(glob.glob('*' + channel_info['struct_channel']))

    if options['final_plot']:
        variables['groups'], variables['refframes'] = \
            GUI.get_plot_var(filenames)

    # instance of the analysis class for each image
    images = [analysis.ImageColocalisation(fileno, options, variables, channel_info)
              for fileno in range(0, len(filenames))]
    logging.info('Finished. Total time taken: %s', datetime.now() - start_time)

    analysis.all_images(images, options, variables, direc)

    plt.show()


