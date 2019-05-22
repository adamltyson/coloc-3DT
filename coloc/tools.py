"""
Adam Tyson | adamltyson@gmail.com | 2019-05-22

"""

import numpy as np
from scipy import ndimage
import pandas as pd
from datetime import datetime
from skimage.external.tifffile import imsave
import logging
import fnmatch
import os.path


def setup_logging(filename, print_level='INFO', file_level='DEBUG'):
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, file_level))

    formatter = logging.Formatter('%(asctime)s - %(levelname)s'
                                  ' - %(processName)s %(filename)s:%(lineno)s'
                                  ' - %(message)s')
    formatter.datefmt = '%Y-%m-%d %H:%M:%S %p'
    fh = logging.FileHandler(
        datetime.now().strftime(filename+'_%Y-%m-%d_%H-%M.logging.txt'))

    fh.setLevel(getattr(logging, file_level))
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, print_level))
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logging.info('Begin')


def bleach_correction_blind(im):
    bleaching = np.mean(im, (1, 2, 3))
    bleach_corrected = im/bleaching[:, None, None, None]
    return bleach_corrected, bleaching


def smooth(im, kernel_width, anisotropy):
    kernel_width_z = anisotropy_adjust(kernel_width, anisotropy, floor=1)
    for t in range(0, len(im)):
        im[t] = ndimage.filters.gaussian_filter(
            im[t], (kernel_width, kernel_width, kernel_width_z))
    return im


def get_colocalisation(im, mask):
    # measures colocalisation (mean intensity) of a marker in a mask
    # also returns total level of marker in mask and vol of mask
    logging.debug('Measuring colocalisation')
    colocalisation = np.zeros(len(im))
    total_fluor = np.zeros(len(im))
    for t in range(0, len(im)):
        im_tmp = im[t]
        mask_tmp = mask[t]
        colocalisation[t] = np.mean(im_tmp[mask_tmp])
        total_fluor[t] = np.sum(im_tmp[mask_tmp])
    return colocalisation, total_fluor


class SldLogParse:
    # parse specific metadata from slidebook exported logs
    def __init__(self, file_path, fourD=True):
        logging.debug('Reading metadata: %s', file_path)
        text = open(file_path, 'r')
        self.z_planes = self.get_var_sld_log(
            text, 'Z Planes')
        self.channels = self.get_var_sld_log(
            text, 'Channels')
        self.mic_per_pix = self.get_var_sld_log(
            text, 'Microns Per Pixel')
        self.z_step = self.get_var_sld_log(
            text, 'Z Step Size Microns')

        if fourD:
            # self.time_points = self.get_var_sld_log(
            #     text, 'Time Points')
            self.time_points = None
            self.ave_timelapse_int = self.get_var_sld_log(
                text, 'Average Timelapse Interval')
        else:
            self.time_points = 1
            self.ave_timelapse_int = []

    @staticmethod
    def get_var_sld_log(text, var_string):
        logging.debug('Reading: %s', var_string)
        val = None
        while True:
            line = text.readline()
            if var_string in line:
                for t in line.split():
                    try:
                        val = float(t) # maybe change
                        break  # if finds a number, stop
                    except ValueError:  # if float doesnt work - try w next
                        pass
                return val


def anisotropy_adjust(param, anisotropy, floor=1):
    if anisotropy is not None:
        param_adjust = np.round(param*anisotropy)
        if param_adjust < floor:
            param_adjust = floor
        return param_adjust
    else:
        return param


def loop_reslice_4d(im, anisotropy, reslice_order=1):
    logging.debug('Reslicing')
    resliced = np.zeros((im.shape[0], round(im.shape[1] * anisotropy),
                         im.shape[2], im.shape[3]))

    for t in range(0, len(im)):
        resliced[t] = ndimage.zoom(im[t], (anisotropy, 1, 1),
                                   order=reslice_order)

    return resliced


def save_results(images, heading,  save_attr):
    # saves properties of images to a csv, with images.heading as heading
    # *args must match the attribute of the class
    logging.info('Saving results to .csv')

    for attr in save_attr:
        df = pd.DataFrame()
        for i in range(0, len(images)):
            df_new = pd.DataFrame(
                {getattr(images[i], heading): getattr(images[i], attr)})
            df = pd.concat([df, df_new], axis=1)  # concat because diff lengths
        df.to_csv(attr+'.csv', encoding='utf-8', index=False)
        del df


def options_variables_write(options, variables, direc):
    write_file = os.path.join(direc, "analysis_options.txt")
    with open(write_file, 'w') as file:
        file.write('directory: ' + direc + '\n')
        file.write('Analysis carried out: ' +
                   datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '\n')

        file.write('\n\n**************  OPTIONS **************\n\n')

        for key, value in options.items():
            file.write('%s: %s\n' % (key, value))

        file.write('\n\n**************  VARIABLES **************\n\n')

        for key, value in variables.items():
            file.write('%s: %s\n' % (key, value))

        file.close()


def im_save_4d(img, fileout, xy_res=1, z_spacing=1):
    # resolution must be in pixels per micron
    # saves a ZYX volume for ImageJ import

    img = np.float32(img)
    # reshape to save as ImageJ TZCYXS format
    img = np.expand_dims(img, axis=2)

    metadata_dict = dict(axes='TZCYX',
                         unit='um',
                         spacing=z_spacing)

    imsave(fileout, img, imagej=True, resolution=(xy_res, xy_res),
           software='tifffile.py', metadata=metadata_dict)


def gen_not_mask(mask, mask2, erosion=None, anisotropy=None):
    """
    "masks" a mask
    returns mask where mask2=False
    option to erode mask too, with anisotropy correction
    """

    mask = np.logical_xor(mask, mask2)
    if erosion is not None:
        if anisotropy is not None:
            erosion_z = anisotropy_adjust(erosion, anisotropy, floor=1)
        else:
            erosion_z = erosion

        erosion = int(erosion)
        erosion_z = int(erosion_z)

        struct = np.ones((erosion, erosion, erosion_z))
        for t in range(0, len(mask)):
            mask[t] = ndimage.binary_erosion(mask[t], struct)

    return mask


def get_match_class_att(obj, strings):
    """
    Takes an object and a list of strings.
    Returns object attributes containing those strings
    :param obj: Any object
    :param strings: List of N strings to match against attributes
    :return matching: the attributes matching args
    """
    matching = []
    class_attributes = dir(obj)

    for string in strings:
        matching.extend(fnmatch.filter(class_attributes, string))

    return matching
