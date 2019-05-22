"""
Adam Tyson | adamltyson@gmail.com | 2019-05-22

"""

import coloc.tools as tools
import coloc.plot as plot
import glob
from skimage import io
import numpy as np
import logging
import matplotlib.pyplot as plt
import skimage.filters
import warnings
import skimage.morphology
from scipy import ndimage


class ImageColocalisation:
    def __init__(self, file_no, options, variables, channel_info):
        self.options = options
        self.variables = variables
        self.channel_info = channel_info
        self.file_no = file_no
        self.functional_anisotropy = []
        self.sum_marker_voxels = []
        self.sum_marker_um3 = []
        self.filename = []
        self.metadata = []
        self.anisotropy = []
        self.marker_thresh = []
        self.marker_im_raw = []
        self.marker_background_im = []
        self.whole_cell = []
        self.fourD = True  # for metadata parse later
        # [*dict] operator unpacks the keys
        self.channel_keys = [*self.channel_info]
        for channel in self.channel_info:
            setattr(self, channel, [])

        self.analysis_run()
        self.mem_cleanup()

    def analysis_run(self):
        self.load(self.file_no, self.channel_info,
                  max_frames=self.variables['max_frames'])

        if self.struct_channel.shape[0] == 1:
            self.fourD = False

        self.metadata, self.anisotropy =\
            get_metadata(self.file_no, fourD=self.fourD)

        self.marker_im_raw = self.struct_channel

        self.reslice(self.options, self.variables, self.channel_keys)

        self.marker_thresh, self.marker_background_im, self.whole_cell =\
            self.segment(self.variables, self.options, self.struct_channel)

        for coloc_im in self.channel_keys:
            if coloc_im is not 'struct_channel':
                coloc_tmp, total_in_marker_tmp = tools.get_colocalisation(
                    getattr(self, coloc_im),
                    getattr(self, 'marker_thresh'))

                coloc_tmp = colocalisation_correct(
                    coloc_tmp, getattr(self, coloc_im),
                    self.marker_background_im, self.whole_cell, self.options)

                setattr(self, coloc_im + "_colocalisation", coloc_tmp)
                setattr(self, coloc_im + "_total_in_marker",
                        total_in_marker_tmp)

        self.get_vol_segmented()

    def mem_cleanup(self):
        logging.debug('Cleaning up memory')
        delattr(self, 'whole_cell')
        for channel in self.channel_info:
            delattr(self, channel)

    def load(self, file_i, channel_info, max_frames=0):
        filenames = sorted(glob.glob('*' + channel_info['struct_channel']))
        file_template_structure = filenames[file_i]

        for channel in channel_info:
            file_template_channel = file_template_structure.replace(
                channel_info['struct_channel'], channel_info[channel])
            file_template_channel = \
                file_template_channel.replace("_T0_", "_T*_")
            file_template_channel = \
                file_template_channel.replace("_T00_", "_T*_")

            filenames_struct = sorted(glob.glob(file_template_channel))

            if max_frames == 0:
                len_t = len(filenames_struct)
            else:
                len_t = max_frames

            tmpim_3d = 1.0 * io.imread(filenames_struct[0], plugin='pil')
            len_z, len_y, len_x = tmpim_3d.shape
            logging.info('Loading: %s', filenames_struct[0])

            if channel is 'struct_channel':
                self.filename = filenames[file_i]
                logging.debug('Calculated from images: %s timepoints, '
                              '%s z-planes, %s x %s pixels',
                              len_t, len_z, len_x, len_y)

            tmpim_4d = np.empty((len_t, len_z, len_y, len_x))
            tmpim_4d[0] = tmpim_3d

            for t in range(1, len_t):
                tmpim_4d[t] =\
                    1.0 * io.imread(filenames_struct[t], plugin='pil')

            setattr(self, channel, tmpim_4d)

    def reslice(self, options, variables, image_list):
        if self.anisotropy is not None:
            if options['reslice']:
                warnings.filterwarnings('ignore', '.*output shape of zoom.*')

                for image in image_list:
                    reslice_tmp = tools.loop_reslice_4d(
                        getattr(self, image), self.anisotropy,
                        reslice_order=variables['reslice_order'])

                    setattr(self, image, reslice_tmp)

                self.metadata.z_planes_resliced =\
                    getattr(self, image_list[0]).shape[1]
                self.functional_anisotropy = 1
                self.metadata.voxel_vol =\
                    self.metadata.voxel_vol*self.metadata.z_planes / \
                    self.metadata.z_planes_resliced
            else:
                self.functional_anisotropy = self.anisotropy

        else:
            logging.error('No metadata for reslicing, skipping ...')

            self.functional_anisotropy = 1

        logging.debug('Image size for processing: '
                      '%s timepoints, %s z-planes, %s x %s pixels',
                      getattr(self, image_list[0]).shape[0],
                      getattr(self, image_list[0]).shape[1],
                      getattr(self, image_list[0]).shape[3],
                      getattr(self, image_list[0]).shape[2])

    def segment(self, variables, options, marker_im):
        logging.debug('starting segmentation')

        marker_im_thresh, marker_background_trace,\
            marker_non_cell_trace, marker_bleaching =\
            marker_segment(marker_im,
                           kernel_width=variables['marker_smooth'],
                           thresh_adj=variables['marker_thresh'],
                           open_dist=variables['marker_noise_rem'],
                           anisotropy=self.anisotropy,
                           bleach_corr=options['bleach_correct'],
                           seg_label='Marker')

        logging.debug('Dilating marker to get background')
        marker_background_im = np.array(marker_im_thresh)
        marker_background_margin = np.array(marker_im_thresh)

        struct_dilate = np.ones((variables['marker_dilate_dist'],
                                 variables['marker_dilate_dist'],
                                 variables['marker_dilate_dist']))
        margin_dilate = np.ones((variables['marker_dilate_margin'],
                                 variables['marker_dilate_margin'],
                                 variables['marker_dilate_margin']))

        for t in range(0, marker_im.shape[0]):
            marker_background_im[t] =\
                ndimage.binary_dilation(marker_im_thresh[t], struct_dilate)
            marker_background_margin[t] =\
                ndimage.binary_dilation(marker_im_thresh[t], margin_dilate)
        whole_cell = np.array(marker_background_im)
        marker_background_im = marker_background_im ^ marker_background_margin

        return marker_im_thresh, marker_background_im, whole_cell

    def get_vol_segmented(self):
        # number of voxels segmented, and volume in  in um^3
        self.sum_marker_voxels = np.zeros(len(self.marker_thresh))
        for t in range(0, len(self.marker_thresh)):
            self.sum_marker_voxels[t] = np.sum(self.marker_thresh[t])

        if self.metadata is not None:
            self.sum_marker_um3 = self.sum_marker_voxels *\
                                  self.metadata.voxel_vol
        else:
            self.sum_marker_um3 = None


def colocalisation_correct(colocalisation, raw_image, non_marker_mask,
                           whole_cell_mask, options):
    if options['bleach_correct']:
        logging.debug('Correcting colocalisation for whole-cell bleaching')
        bleaching, tmp1 = \
            tools.get_colocalisation(raw_image, whole_cell_mask)
        colocalisation = colocalisation / bleaching

    if options['enrichment']:
        logging.debug('Normalising for enrichment')
        background_fluor, tmp1 = \
            tools.get_colocalisation(raw_image, non_marker_mask)
        colocalisation = colocalisation / background_fluor

    if options['normalise']:
        logging.debug('Normalising results')
        colocalisation = np.divide(colocalisation, colocalisation[0])

    return colocalisation


def all_images(images, options, variables, direc):
    plot_size = variables['plot_size']
    if options['save_vars']:
        logging.info('Saving analysis options')
        tools.options_variables_write(options, variables, direc)

    if options['save_results']:
        save_attrs = tools.get_match_class_att(
            images[0], variables['save_attrs'])
        tools.save_results(images, variables['save_heading'], save_attrs)

    if options['save_seg']:
        logging.info('Saving segmentation as 4D tiff')

        for image in images:
            prefix = image.filename.rsplit('.tif')[0]  # remove .tif
            xy_res = 1 / image.metadata.mic_per_pix
            z_spacing = image.metadata.z_step

            if options['reslice']:
                z_spacing = z_spacing/image.anisotropy

            fileout = prefix + '_marker_background.tif'
            tools.im_save_4d(image.marker_background_im,
                             fileout,
                             xy_res=xy_res,
                             z_spacing=z_spacing)

            fileout = prefix + '_marker_seg.tif'
            tools.im_save_4d(image.marker_thresh,
                             fileout,
                             xy_res=xy_res,
                             z_spacing=z_spacing)

    if options['seg_plot'] or options['final_plot']:
        plt.switch_backend('Qt5Agg')
    if options['seg_plot']:
        logging.info('Plotting segmentation')
        warnings.filterwarnings(
            'ignore', '.*Attempting to set identical left==right results.*')
        for image in images:
            plot.scroll_overlay_projection(
                image.marker_im_raw,
                image.marker_thresh,
                title='Marker analysis - '
                      'marker segmentation (z projection)',
                figsize=(4*plot_size, 5*plot_size))

    if options['final_plot']:
        if images[0].fourD:
            plot_attrs = tools.get_match_class_att(
                images[0], variables['summary_plot_attrs'])
            plot.mean_sd_plot(images, variables['groups'],
                              variables['refframes'], plot_attrs)
        else:
            logging.info('Only one timepoint, skipping final plot.')

    plt.show()


def parse_metadata(file_no, fourD=True):
    metadata_files = glob.glob('*.log')
    path = metadata_files[file_no]
    metadata = tools.SldLogParse(path, fourD=fourD)

    logging.debug('Metadata from log file: %s timepoints, %s z-planes, '
                  '%s microns per pixel, %s microns per z step, '
                  '%s channels, timelapse interval: %s ms',
                  metadata.time_points,
                  metadata.z_planes,
                  metadata.mic_per_pix,
                  metadata.z_step,
                  metadata.channels,
                  metadata.ave_timelapse_int)

    return metadata


def get_metadata(filename, fourD=True):
    try:
        metadata = parse_metadata(filename, fourD=fourD)
        metadata.voxel_vol = metadata.z_step * np.square(metadata.mic_per_pix)
        anisotropy = metadata.z_step/metadata.mic_per_pix
        logging.debug('Anisotropy factor: %s', np.round(anisotropy, 2))
    except OSError:
        metadata = None
        anisotropy = None

        logging.error('Metadata not found,assuming isotropic voxels')

    return metadata, anisotropy


def marker_segment(im, kernel_width=1, thresh_adj=1, open_dist=1,
                   anisotropy=None, bleach_corr=False, seg_label=''):
    """
    INPUT:
        im: raw image to be segmented
        kernel_width: Gaussian smooth sigma
        thresh_adj: otsu threshold adjustment multiplier
        open_dist: binary opening distance
        anisotropy: anisotropy factor in z (dim 1 - TZYX) to adjust sigma
        bleach_corr: logical - correct for bleaching or not
        seg_label: label to add to logging statements

    OUTPUT:
        im_thresh: segmented image
        mean_intensity: mean intensity over time within the segmented object
        non_marker_mean_intensity: mean intensity over time outside the object
        bleaching_trace: mean intensity across whole image
    """

    bleaching_trace = None
    im_orig = np.array(im)
    if bleach_corr:
        logging.debug('%s - bleach correction', seg_label)
        im, bleaching_trace = tools.bleach_correction_blind(im)

    if kernel_width is not 0:
        logging.debug('%s - smoothing', seg_label)
        im = tools.smooth(im, kernel_width, anisotropy)

    logging.debug('%s - thresholding', seg_label)
    im_otsu = thresh_adj * skimage.filters.threshold_otsu(im)
    im_thresh = im > im_otsu

    if open_dist is not 0:
        logging.debug('%s - noise (small object) removal', seg_label)
        struct = np.ones((open_dist, open_dist, open_dist))
        for t in range(0, im.shape[0]):
            im_thresh[t] = ndimage.binary_opening(im_thresh[t], struct)

    logging.debug('%s - calculating bleaching', seg_label)
    mean_intensity = np.zeros(len(im))
    non_marker_mean_intensity = np.zeros(len(im))

    for t in range(0, len(im)):
        im_tmp = im_orig[t]
        im_thresh_tmp = im_thresh[t]
        mean_intensity[t] = np.mean(im_tmp[im_thresh_tmp])
        non_marker_mean_intensity[t] = np.mean(im_tmp[~im_thresh_tmp])

    return im_thresh, mean_intensity,\
        non_marker_mean_intensity, bleaching_trace

