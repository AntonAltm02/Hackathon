
# DDSM Mammography
## Summary
This dataset consists of images from the DDSM [1] and CBIS-DDSM [3] datasets. The images have been pre-processed and 
converted to 299x299 images by extracting the ROIs. The data is stored as tfrecords files for TensorFlow.

The dataset contains 55,890 training examples, of which 14% are positive and the remaining 86% negative, 
divided into 5 tfrecords files.

## Note 
The data has been separated into training and test as per the division in the CBIS-DDSM dataset. The test files have 
been divided equally into test and validation data. However, the split between test and validation
data was done incorrectly, resulted in the test numpy files containing only masses and the validation files containing 
only calcifications. These files should be combined in order to have balanced and complete test data.

## Pre-processing
The dataset consists of negative images from the DDSM dataset and positive images from the CBIS-DDSM dataset. 
The data was pre-processed to convert it into 299x299 images. The negative (DDSM) images were tiled into 598x598 tiles, which were then resized to 299x299.

The positive (CBIS-DDSM) images had their ROIs extracted using the masks with a small amount of padding to 
provide context. Each ROI was then randomly cropped three times into 598x598 images, with random flips and 
rotations, and then the images were resized down to 299x299.

The images are labeled with two labels:
+ label_normal - 0 for negative and 1 for positive 
+ label - full multi-class labels, 0 is negative, 1 is benign calcification, 2 is benign mass, 3 is malignant calcification, 4 is malignant mass

[1] The Digital Database for Screening Mammography, Michael Heath, Kevin Bowyer, Daniel Kopans, Richard Moore and 
W. Philip Kegelmeyer, in Proceedings of the Fifth International Workshop on Digital Mammography, M.J. Yaffe, ed., 
212-218, Medical Physics Publishing, 2001. ISBN 1-930524-00-5.
[2] Rebecca Sawyer Lee, Francisco Gimenez, Assaf Hoogi , Daniel Rubin (2016). Curated Breast Imaging Subset of DDSM. The Cancer Imaging Archive.