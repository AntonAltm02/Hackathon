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

#### DDSM - Digital Database for Screening Mammography

This dataset consists of images from the DDSM and CBIS-DDSM datasets. The images have been pre-processed and converted to 299x299 images by extracting the ROIs. The data is stored as tfrecords files for TensorFlow.
The dataset contains 55,890 training examples, of which 14% are positive and the remaining 86% negative, divided into 5 tfrecords files. The data has been separated into training and test as per the division in the CBIS-DDSM dataset. The test files have been divided equally into test and validation data. However the split between test and validation data was done incorrectly, resulted in the test numpy files containing only masses and the validation files containing only calcifications. These files should be combined in order to have balanced and complete test data.

The dataset consists of negative images from the DDSM dataset and positive images from the CBIS-DDSM dataset. The data was pre-processed to convert it into 299x299 images. The negative (DDSM) images were tiled into 598x598 tiles, which were then resized to 299x299. The positive (CBIS-DDSM) images had their ROIs extracted using the masks with a small amount of padding to provide context. Each ROI was then randomly cropped three times into 598x598 images, with random flips and rotations, and then the images were resized down to 299x299.

The images are labeled with two labels:
+ label_normal - 0 for negative and 1 for positive
+ label - full multi-class labels, 0 is negative, 1 is benign calcification, 2 is benign mass, 3 is malignant calcification, 4 is malignant mass

Dataset: https://www.kaggle.com/datasets/skooch/ddsm-mammography

#### CBIS-DDSM - Curated Breast Imaging Subset of DDSM:

This CBIS-DDSM (Curated Breast Imaging Subset of DDSM) is an updated and standardized version of the Digital Database for Screening Mammography (DDSM). The DDSM is a database of 2,620 scanned film mammography studies. It contains normal, benign, and malignant cases with verified pathology information. The scale of the database along with ground truth validation makes the DDSM a useful tool in the development and testing of decision support systems. The CBIS-DDSM collection includes a subset of the DDDSM data selected and curated by a trained mammographer. The images have been decompressed and converted to DICOM format. Updated ROI segmentation and bounding boxes, and pathologic diagnosis for training data are also included (https://www.nature.com/articles/sdata2017177).

Please note that the image data for this collection is structured such that each participant has multiple patient IDs. For example, participant 00038 has 10 separate patient IDs which provide information about the scans within the IDs (e.g. Calc-Test_P_00038_LEFT_CC, Calc-Test_P_00038_RIGHT_CC_1). This makes it appear as though there are 6,671 patients according to the DICOM metadata, but there are only 1,566 actual participants in the cohort.
Dataset: https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset?resource=download
Notebook: https://www.kaggle.com/code/baselanaya/breast-cancer-detection-using-cnn/notebook

#### A Multi-million Mammography Image Dataset and Population-Based Screening Cohort for the Training and Evaluation of Deep Neural Networks-the Cohort of Screen-Aged Women (CSAW)

For AI researchers, access to a large and well-curated dataset is crucial. Working in the field of breast radiology, our aim was to develop a high-quality platform that can be used for evaluation of networks aiming to predict breast cancer risk, estimate mammographic sensitivity, and detect tumors. Our dataset, Cohort of Screen-Aged Women (CSAW), is a population-based cohort of all women 40 to 74 years of age invited to screening in the Stockholm region, Sweden, between 2008 and 2015. All women were invited to mammography screening every 18 to 24 months free of charge. Images were collected from the PACS of the three breast centers that completely cover the region. DICOM metadata were collected together with the images. Screening decisions and clinical outcome data were collected by linkage to the regional cancer center registers. Incident cancer cases, from one center, were pixel-level annotated by a radiologist. A separate subset for efficient evaluation of external networks was defined for the uptake area of one center. The collection and use of the dataset for the purpose of AI research has been approved by the Ethical Review Board. CSAW included 499,807 women invited to screening between 2008 and 2015 with a total of 1,182,733 completed screening examinations. Around 2 million mammography images have currently been collected, including all images for women who developed breast cancer. There were 10,582 women diagnosed with breast cancer; for 8463, it was their first breast cancer. Clinical data include biopsy-verified breast cancer diagnoses, histological origin, tumor size, lymph node status, Elston grade, and receptor status. One thousand eight hundred ninety-one images of 898 women had tumors pixel level annotated including any tumor signs in the prior negative screening mammogram. Our dataset has already been used for evaluation by several research groups. We have defined a high-volume platform for training and evaluation of deep neural networks in the domain of mammographic imaging (https://pubmed.ncbi.nlm.nih.gov/31520277/).

#### VinDr-Mammo: A large-scale benchmark dataset for computer-aided diagnosis in full-field digital mammography

Mammography, or breast X-ray imaging, is the most widely used imaging modality to detect cancer and other breast diseases. Recent studies have shown that deep learning-based computer-assisted detection and diagnosis (CADe/x) tools have been developed to support physicians and improve the accuracy of interpreting mammography. A number of large-scale mammography datasets from different populations with various associated annotations and clinical data have been introduced to study the potential of learning-based methods in the field of breast radiology. With the aim to develop more robust and more interpretable support systems in breast imaging, we introduce VinDr-Mammo, a Vietnamese dataset of digital mammography with breast-level assessment and extensive lesion-level annotations, enhancing the diversity of the publicly available mammography data. The dataset consists of 5,000 mammography exams, each of which has four standard views and is double read with disagreement (if any) being resolved by arbitration. The purpose of this dataset is to assess Breast Imaging Reporting and Data System (BI-RADS) and breast density at the individual breast level. In addition, the dataset also provides the category, location, and BI-RADS assessment of non-benign findings. We make VinDr-Mammo publicly available as a new imaging resource to promote advances in developing CADe/x tools for mammography interpretation (https://pubmed.ncbi.nlm.nih.gov/31520277/).

#### BC-MRI-SEG - A Breast Cancer MRI Tumor Segmentation Benchmark:

Binary breast cancer tumor segmentation with Magnetic Resonance Imaging (MRI) data is typically trained and evaluated 
on private medical data, which makes comparing deep learning approaches difficult. We propose a benchmark (BC-MRI-SEG) 
for binary breast cancer tumor segmentation based on publicly available MRI datasets. The benchmark consists of four 
datasets in total, where two datasets are used for supervised training and evaluation, and two are used for zero-shot 
evaluation. Additionally we compare state-of-the-art (SOTA) approaches on our benchmark and provide an exhaustive list 
of available public breast cancer MRI datasets (https://arxiv.org/html/2404.13756v1#S3)
Dataset Links: 
- RIDER: https://wiki.cancerimagingarchive.net/display/Public/RIDER+Breast+MRI
- BreastDM: https://github.com/smallboy-code/Breast-cancer-dataset
- ISPY1: https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=101942541#101942541215b684587f64c8cab1ffc45cd63f339
- DUKE: https://www.cancerimagingarchive.net/collection/duke-breast-cancer-mri/

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

### Image Classifcation Model:
#### EfficientNetV2:

EfficientNetV2, a new family of convolutional networks, have faster
training speed and better parameter efficiency  than previous models. To develop these models, a combination of 
training-aware neural architecture search and scaling is used, to jointly optimize
training speed and parameter efficiency. The models were searched from the search space enriched
with new ops such as Fused-MBConv. Our experiments show that EfficientNetV2 models train
much faster than state-of-the-art models while
being up to 6.8x smaller.
Our training can be further sped up by progressively increasing the image size during training,
but it often causes a drop in accuracy. To compensate for this accuracy drop, we propose an
improved method of progressive learning, which adaptively adjusts regularization (e.g. data augmentation) along with image size.
With progressive learning, our EfficientNetV2 significantly outperforms previous models on ImageNet and CIFAR/Cars/Flowers datasets. By
pretraining on the same ImageNet21k, our EfficientNetV2 achieves 87.3% top-1 accuracy on
ImageNet ILSVRC2012, outperforming the recent ViT by 2.0% accuracy while training 5x-11x
faster using the same computing resources.


## Results
