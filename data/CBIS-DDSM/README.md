
# CBIS-DDSM - Curated Breast Imaging Subset of DDSM:
## Summary
This CBIS-DDSM (Curated Breast Imaging Subset of DDSM) is an updated and standardized version of the Digital 
Database for Screening Mammography (DDSM). The DDSM is a database of 2,620 scanned film mammography studies. It contains 
normal, benign, and malignant cases with verified pathology information. The scale of the database along with ground 
truth validation makes the DDSM a useful tool in the development and testing of decision support systems. The CBIS-DDSM 
collection includes a subset of the DDDSM data selected and curated by a trained mammographer. The images have been 
decompressed and converted to DICOM format. Updated ROI segmentation and bounding boxes, and pathologic diagnosis for 
training data are also included (https://www.nature.com/articles/sdata2017177).

Please note that the image data for this collection is structured such that each participant has multiple patient IDs. 
For example, participant 00038 has 10 separate patient IDs which provide information about the scans within the IDs 
(e.g. Calc-Test_P_00038_LEFT_CC, Calc-Test_P_00038_RIGHT_CC_1). This makes it appear as though there are 6,671 patients 
according to the DICOM metadata, but there are only 1,566 actual participants in the cohort.

## Citations & Data Usage Policy
Users of this data must abide by the TCIA Data Usage Policy and the Creative Commons Attribution 3.0 Unported License 
under which it has been published. Attribution should include references to the following citations:
Rebecca Sawyer Lee, Francisco Gimenez, Assaf Hoogi , Daniel Rubin  (2016). Curated Breast Imaging Subset of DDSM [Dataset]. 
The Cancer Imaging Archive. DOI: https://doi.org/10.7937/K9/TCIA.2016.7O02S9CY