# Breast Cancer Dual-Model Prediction

## Overview
This repository hosts a dual-model approach for breast cancer research, focusing on two key tasks:
+ Zero-shot single-cell data prediction: Classifying cells as cancerous or non-cancerous without prior labeled data.
+ Breast cancer region segmentation: Identifying labeled cancerous areas in MRI or mammography images.

## Features
+ Zero-shot learning: Utilizes advanced AI techniques to classify single-cell data without explicit training labels.
+ Medical image segmentation: Implements deep learning models to detect breast cancer regions in MRI/mammography scans.
+ Multi-modal approach: Combines single-cell analysis and image processing for a comprehensive cancer detection framework.

## Dataset

Single-cell data: [Dataset source or link]

### Medical images:
#### CBIS-DDSM - Curated Breast Imaging Subset of DDSM:

This CBIS-DDSM (Curated Breast Imaging Subset of DDSM) is an updated and standardized version of the Digital Database 
for Screening Mammography (DDSM). The DDSM is a database of 2,620 scanned film mammography studies. It contains normal, 
benign, and malignant cases with verified pathology information. The scale of the database along with ground truth 
validation makes the DDSM a useful tool in the development and testing of decision support systems. The CBIS-DDSM 
collection includes a subset of the DDDSM data selected and curated by a trained mammographer. The images have been 
decompressed and converted to DICOM format. Updated ROI segmentation and bounding boxes, and pathologic diagnosis for 
training data are also included. A manuscript describing how to use this dataset in detail is available at 
https://www.nature.com/articles/sdata2017177.

Published research results from work in developing decision support systems in mammography are difficult to replicate 
due to the lack of a standard evaluation data set; most computer-aided diagnosis (CADx) and detection (CADe) algorithms 
for breast cancer in mammography are evaluated on private data sets or on unspecified subsets of public databases. 
Few well-curated public datasets have been provided for the mammography community. These include the DDSM, the 
Mammography Imaging Analysis Society (MIAS) database, and the Image Retrieval in Medical Applications (IRMA) project. 
Although these public data sets are useful, they are limited in terms of data set size and accessibility.

For example, most researchers using the DDSM do not leverage all its images for a variety of historical reasons. 
When the database was released in 1997, computational resources to process hundreds or thousands of images were not 
widely available. Additionally, the DDSM images are saved in non-standard compression files that require the use of 
decompression code that has not been updated or maintained for modern computers. Finally, the ROI annotations for the 
abnormalities in the DDSM were provided to indicate a general position of lesions, but not a precise segmentation for 
them. Therefore, many researchers must implement segmentation algorithms for accurate feature extraction. This causes 
an inability to directly compare the performance of methods or to replicate prior results. The CBIS-DDSM collection 
addresses that challenge by publicly releasing a curated and standardized version of the DDSM for evaluation of future 
CADx and CADe systems (sometimes referred to generally as CAD) research in mammography.

Please note that the image data for this collection is structured such that each participant has multiple patient IDs. 
For example, participant 00038 has 10 separate patient IDs which provide information about the scans within the IDs 
(e.g. Calc-Test_P_00038_LEFT_CC, Calc-Test_P_00038_RIGHT_CC_1). This makes it appear as though there are 6,671 patients 
according to the DICOM metadata, but there are only 1,566 actual participants in the cohort.
Dataset: https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset?resource=download
Notebook: https://www.kaggle.com/code/baselanaya/breast-cancer-detection-using-cnn/notebook

#### BC-MRI-SEG - A Breast Cancer MRI Tumor Segmentation Benchmark:

Binary breast cancer tumor segmentation with Magnetic Resonance Imaging (MRI) data is typically trained and evaluated 
on private medical data, which makes comparing deep learning approaches difficult. We propose a benchmark (BC-MRI-SEG) 
for binary breast cancer tumor segmentation based on publicly available MRI datasets. The benchmark consists of four 
datasets in total, where two datasets are used for supervised training and evaluation, and two are used for zero-shot 
evaluation. Additionally we compare state-of-the-art (SOTA) approaches on our benchmark and provide an exhaustive list 
of available public breast cancer MRI datasets. 
Dataset Links: 
- RIDER: https://wiki.cancerimagingarchive.net/display/Public/RIDER+Breast+MRI
- BreastDM: https://github.com/smallboy-code/Breast-cancer-dataset
- ISPY1: https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=101942541#101942541215b684587f64c8cab1ffc45cd63f339
- DUKE: https://www.cancerimagingarchive.net/collection/duke-breast-cancer-mri/

Dataset: https://arxiv.org/html/2404.13756v1#S3


## Model Architecture

### Single-cell classifier: 

### Image segmentation model:
#### DC-UNet - Rethinking the U-Net Architecture with Dual Channel Efficient CNN for Medical Images Segmentation:

Recently, deep learning has become much more popular in computer vision area. The Convolution Neural Network (CNN) has 
brought a breakthrough in images segmentation areas, especially, for medical images. In this regard, U-Net is the 
predominant approach to medical image segmentation task. The U-Net not only performs well in segmenting multimodal 
medical images generally, but also in some tough cases of them. However, we found that the classical U-Net architecture
has limitation in several aspects. Therefore, we applied modifications: 1) designed efficient CNN architecture to 
replace encoder and decoder, 2) applied residual module to replace skip connection between encoder and decoder to 
improve based on the-state-of-the-art U-Net model. Following these modifications, we designed a novel 
architecture--DC-UNet, as a potential successor to the U-Net architecture. We created a new effective CNN architecture 
and build the DC-UNet based on this CNN. We have evaluated our model on three datasets with tough cases and have 
obtained a relative improvement in performance of 2.90%, 1.49% and 11.42% respectively compared with classical U-Net. 
In addition, we used the Tanimoto similarity to replace the Jaccard similarity for gray-to-gray image comparisons.
Paper: https://arxiv.org/abs/2006.00414
GitHub: https://github.com/AngeLouCN/DC-UNet

## Results
