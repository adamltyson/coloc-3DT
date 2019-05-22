## coloc-3DT
##### Adam Tyson | 2019-05-22 | adamltyson@gmail.com 

#### Returns colocalisation of N channels in another, segmented channel


Written for data acquired in [Slidebook](https://www.intelligent-imaging.com/slidebook), 
and exported as [OME-Tiff](https://docs.openmicroscopy.org/ome-model/5.6.3/ome-tiff/), 
but any multipage TIFF should work if the filenames match the 
hardcoded pattern.

Tested on Windows (7 & 10), macOS (10.14) and Ubuntu (18.04) with Python 3.5, 3.6 & 3.7.

##### Requirements
    Python >= 3.5
    matplotlib
    scikit-image
    pillow
    numpy
    tk
    scipy
    pandas
    seaborn
	
## Instructions (install):
1. Download [anaconda](https://www.anaconda.com/download/) or [miniconda](https://conda.io/miniconda.html)
2. Set up a conda environment (e.g. by opening "Anaconda Prompt") and run:
    ```bash
	    conda create --name coloc-3DT numpy scipy pandas pillow scikit-image matplotlib tk seaborn
	    source activate coloc-3DT
	```
3. Clone or download repository (e.g. **Clone or download -> Download ZIP**, then unzip **coloc-3DT-master.zip**)
4. Navigate to `/coloc-3DT` in terminal, or in IDE (e.g. PyCharm, Spyder)
5. Run main program `python main.py`
	
## Instructions (use):

1. Export 3D image as multipage tiff (default if <4GB in Slidebook), one 3D image per channel, per timepoint. All images can be saved into the same directory.
	* Ensure the `.log` file is also generated. This is read to ensure accurate metadata, allthough the analysis will run without it, without reslicing to isotropic resolution and without adjusting the time axes of the final plots.

2. Confirm or change options (the defaults can be changed in `coloc-3DT.options.options_variables`:
	* **Bleach correction?** - correct for bleaching over time. Blind correction for segmentation, then corrects colocalisation levels for mean in-cell intensity
	* **Normalise correlation to baseline (t=0)?** - adjust final colocalisation measures. Normalises to the first timepoint
	* **Measure enrichment?** - Measure levels in marker relative to the surrounding area
	* **Segmentation checker?** - after all analysis is done, shows scroll plots of maximum-intensity projects over time to check segmentation of both channels. Useful when setting up the analysis for the first time. It is useful to check the influence of the various parameters on the segmentation on a representative selection of images.
	* **Plot final results?** - plot graphs of all timecourses (adjusted to the same temporal scale)
	* **Reslice to isotropic resolution?** - use metadata to reslice the data in z to generate data with isotropic voxels. All analysis steps are carried out in 3D, and this prevents bias to a particular dimension
	* **Save results as .csv?** - save all generated parameters to a csv file. One column per movie, one file per parameter
	* **Save analysis options as .txt?** - saves all chosen options and parameters to a .txt file to be reused
	* **Save segmentation as tif?** - save all segmentations as 4D (TZYX), ImageJ-compatible .tif files
	* **Testing?** - unused, to allow various testing options

3. Confirm or change variables (the defaults can be changed in `coloc-3DT.options.options_variables`:
	* **Channel 1 smoothing width (pixels)** smoothing magnitude for channel 1 (marker)
	* **Marker threshold adjustment (a.u)** - arbitrary threshold multiplier. Increase to be more stringent (and vice versa)
	* **Marker noise removal (pixels)** how big can bright spots be and not be classed as a marker
	* **Marker normalisation mask (non-marker) erosion (pixels)** parameter for the `Measure enrichment?` option. This ensures that no voxels inside the marker are classes as background. This final mask isn't shown as part of the segmentation checker, but can be saved as a .tif and inspected
	* **Cell size estimation radius (pixels)** as above, how big should the "background" be (similar to e.g. nuclear size)
	* **Reslicing interpolation order** - parameter for the "fit" of the z-reslicing
	* **Plot size (a.u.)** - arbitrary number to change size of the plots (if using a vastly differently sized screen). All matplotlib plots can be resized anyway
	* **Number colocalisation channels** - how many channels to assess colocalisation
	* **Number of frames to analyse (0 for all)** - if testing, can only analyse N frames. If set to 0, all frames will be analysed. 

4. Choose the channels you want to measure (pre-populated based on the number of colocalisation channels chosen.)
    * Based on suffix `T00_C**channel**.tif`  (can also change file extension)

5. If `Plot final results?` is selected, a `Choose plotting options` window will come up. For each image the `Group` (a text label) and a `Reference frame` (which frame corresponds to the same biological timepoint) can be entered.

The script will then run through all images corresponding to the structural marker string (e.g. '*T0_C0.tif'), and will find all corresponding images for all timepoints. This data will be loaded, the marker segmented and the colocalisation will be measured, and then corrected for the factors selected above. The analysis parameters, segmented images and final results will be saved, and then the plots will be shown. The `segmentation checker` plots will be shown one at a time, close one to see the next. Details about the analysis will be printed and saved as a log file.

## Outputs:
* **coloc-3DT-`date`-`time`.log** - analysis log
* **analysis_options.txt** - all options and variables defined above
* **`channel`_colocalisation.csv** - mean level of the channel in the marker, adjusted with respect to the above parameters
* **`channel`_total_in_marker.csv** - total level of channel in marker (like colocalisation, but sum, and not corrected)
* **sum_marker_um3.csv** - total channel 1 marker volume (in real units)
* **sum_marker_voxels.csv** - total channel 1 marker volume(in voxels, which may have been resliced)
* **`image`_marker_seg.tif** - marker segmentation image
* **`image`_marker_background.tif** - background segmentation image
