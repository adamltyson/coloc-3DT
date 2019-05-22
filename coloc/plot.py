"""
Adam Tyson | adamltyson@gmail.com | 2019-05-22

"""

import matplotlib.pyplot as plt
from random import randint
import numpy as np
from matplotlib.widgets import Slider
import logging
import seaborn as sns
import pandas as pd


def hist(x, plot_style=None):
    if plot_style is not None:
        plt.style.use(plot_style)
    x = np.ndarray.flatten(x)

    plt.hist(x)
    plt.title("Histogram")
    plt.show()


def simple_plot(img, title, plotsize, plot_style=None):
    if plot_style is not None:
        plt.style.use(plot_style)

    plt.figure(figsize=(plotsize, plotsize))
    plt.imshow(img, cmap="Greys_r")
    plt.title(title)
    plt.show(block=False)


def plot_1d(data, plotsize, title=None, plot_style=None):
    if plot_style is not None:
        plt.style.use(plot_style)

    plt.figure(figsize=(3*plotsize, 3*plotsize))
    plt.plot(data)
    if title is not None:
        plt.title(title)
    plt.show(block=False)


def rand_plot_compare(im1, im2, num_cols=10, plotsize=3, title=None,
                      min_val=None, max_val=None, plot_style=None):
    if plot_style is not None:
        plt.style.use(plot_style)

    # Initialize the subplot panels on top of each other
    fig, ax = plt.subplots(nrows=2, ncols=num_cols,
                           figsize=(num_cols*plotsize, 2*plotsize))
    if min_val is None:
        min_val = im1.min()
    if max_val is None:
        max_val = im1.max()

    for plot in range(0, num_cols):
        t = randint(0, im1.shape[0]-1)
        z = randint(0, im1.shape[1]-1)
        ax[0][plot].imshow(im1[t][z], vmin=min_val, vmax=max_val, cmap="Greys_r")
        ax[0][plot].axis('off')
        if plot is 0:
            ax[0][plot].set_title('Raw:   T='+str(t)+', Z='+str(z))
            ax[1][plot].set_title('Thresholded')
        else:
            ax[0][plot].set_title('T='+str(t)+', Z='+str(z))
        ax[1][plot].imshow(im2[t][z], cmap="Greys_r")
        ax[1][plot].axis('off')

    if title is not None:
        fig.suptitle(title)
        plt.show(block=False)


def plot_summary(images, plot_size, plot_style, label, *args):
    # plots properties of images to , with images.labels as figure labels
    # *args must match the attribute of the class
    logging.info('Plotting summary')

    for arg in args:
        legends = []
        if plot_style is not None:
            plt.style.use(plot_style)
        plt.figure(figsize=(plot_size, plot_size))

        for i in range(0, len(images)):
            # assume if one has metadata, they all do
            x = np.arange(0, len(getattr(images[i], arg)))
            if images[0].metadata is not None:
                x = x * round(images[i].metadata.ave_timelapse_int/60000)

            plt.plot(x, getattr(images[i], arg))
            legends.append(getattr(images[i], label))

        plt.legend(legends, loc='upper left')
        plt.title(arg+' over time')

        if images[0].metadata is not None:
            plt.xlabel('Time (minutes)')
        else:
            logging.debug('No temporal metadata available, plotting anyway')
            plt.xlabel('Time (frames)')

        plt.ylabel(arg)

        plt.show(block=False)


def scroll_overlay_projection(im_raw_in, im_seg_in, title=None,
                              figsize=(12, 16), im1_max=None, im2_max=None):
    global im_raw
    global im_seg

    im_raw = np.amax(im_raw_in, 1)
    im_seg = np.amax(im_seg_in, 1)

    if title is None:
        title = "Image - z projection"

    t_min = 0
    t_max = len(im_raw)-1
    t_init = 0

    fig = plt.figure(figsize=figsize)

    im_ax1 = plt.axes([0.1, 0.2, 0.45, 0.65], )
    im_ax2 = plt.axes([0.5, 0.2, 0.45, 0.65], )

    slider_ax = plt.axes([0.1, 0.05, 0.8, 0.05])
    im_ax1.set_xticks([])
    im_ax1.set_yticks([])

    im_ax2.set_xticks([])
    im_ax2.set_yticks([])

    plt.sca(im_ax1)
    im_plot1 = plt.imshow(im_raw[t_init], cmap="Greys_r", vmin=0, vmax=im1_max)

    plt.sca(im_ax2)
    im_plot2 = plt.imshow(im_seg[t_init], cmap="Greys_r", vmin=0, vmax=im2_max)

    t_slider = Slider(slider_ax, 'Timepoint', t_min, t_max, valfmt="%i",
                      valinit=t_init, valstep=1)
    plt.suptitle(title)

    def update(t):
        t = int(t)
        im_plot1.set_data(im_raw[t])
        im_plot2.set_data(im_seg[t])

        fig.canvas.draw_idle()     # redraw the plot

    t_slider.on_changed(update)
    plt.show(block=True)


def mean_sd_plot(images, groups, refframes, attrs, plot_indiv=True):
    sns.set(font_scale=2, rc={"lines.linewidth": 5})
    # get num groups, otherwise plot with "hue" fails with one group
    num_groups = len(set(list(groups.values())))
    logging.debug('plotting summary data')
    for attr in attrs:
        cell_no = 0
        dfcombo = pd.DataFrame(columns=['coloc', 'time', 'group', 'cell'])
        for image in images:
            test = pd.DataFrame({'coloc': getattr(image, attr)})
            test.insert(1, 'cell', cell_no)
            test.insert(1, 'group', groups[image.filename])
            test['time'] = np.around(
                ((np.arange(0, len(getattr(image, attr))) -
                  refframes[image.filename]) *
                 image.metadata.ave_timelapse_int / 60000),
                decimals=2)
            dfcombo = dfcombo.append(test, sort=True)
            cell_no = cell_no + 1

        plt.figure()
        if num_groups is 1:
            ax = sns.lineplot(x="time", y="coloc", data=dfcombo, ci='sd')
        else:
            ax = sns.lineplot(x="time", y="coloc", hue="group", data=dfcombo,
                                ci='sd')
        ax.set(xlabel='Time relative to reference [mins]',
               ylabel='Colocalisation [a.u]')
        handles, _ = ax.get_legend_handles_labels()
        ax.set_title(attr + " mean +/- SD")

        if plot_indiv:
            plt.figure()
            if num_groups is 1:
                ax2 = sns.lineplot(x="time", y="coloc", data=dfcombo,
                                   units="cell", estimator=None, lw=1)
            else:
                ax2 = sns.lineplot(x="time", y="coloc", hue="group",
                                   data=dfcombo, units="cell",
                                   estimator=None, lw=1)
            ax2.set(xlabel='Time relative to reference [mins]',
                    ylabel='Colocalisation [a.u]')
            handles, _ = ax.get_legend_handles_labels()
            ax2.set_title(attr)
