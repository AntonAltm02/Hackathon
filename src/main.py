import os
import pandas as pd
import matplotlib.pyplot as plt
import PIL
import glob

"""
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
Mammographic Imaging Analysis Society (MIAS) database, and the Image Retrieval in Medical Applications (IRMA) project. 
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
"""

if __name__ == '__main__':
    script_dir = os.path.dirname(__file__)
    data_path = os.path.join(script_dir + "/../data")

    dicom_info = pd.read_csv(data_path + "/dicom_info.csv")
    image_dir = data_path + "/jpeg"

    print(dicom_info.head())
    print(dicom_info.info())

    """
    Cropped images
    """
    cropped_images = dicom_info[dicom_info.SeriesDescription == "cropped images"].image_path
    cropped_images.head()
    cropped_images = cropped_images.apply(lambda x: x.replace('CBIS-DDSM/jpeg', image_dir))
    cropped_images.head()
    for file in cropped_images[0:10]:
        cropped_images_show = PIL.Image.open(file)
        gray_img = cropped_images_show.convert("L")
        plt.imshow(gray_img, cmap='gray')

    """
    Full images
    """
    full_images = dicom_info[dicom_info.SeriesDescription == "full mammogram images"].image_path
    full_images.head()
    full_images = full_images.apply(lambda x: x.replace('CBIS-DDSM/jpeg', image_dir))
    full_images.head()
    for file in full_images[0:10]:
        full_images_show = PIL.Image.open(file)
        gray_img = full_images_show.convert("L")
        plt.imshow(gray_img, cmap='gray')

    """
    ROI images
    """
    roi_images = dicom_info[dicom_info.SeriesDescription == "ROI mask images"].image_path
    roi_images.head()
    roi_images = roi_images.apply(lambda x: x.replace('CBIS-DDSM/jpeg', image_dir))
    roi_images.head()
    for file in roi_images[0:10]:
        roi_images_show = PIL.Image.open(file)
        gray_img = roi_images_show.convert("L")
        plt.imshow(gray_img, cmap='gray')

    """
    Calcification and mass data cases
    """
    df_calc_case = pd.read_csv(data_path + "/calc_case_description_train_set.csv")
    df_calc_case.head(5)
    df_mass_case = pd.read_csv(data_path + "/mass_case_description_train_set.csv")
    df_calc_case.head(5)

    """
    Dicom info cleaning
    """
    dicom_info_clean = dicom_info.copy()
    dicom_info_clean.drop(
        ['PatientBirthDate', 'AccessionNumber', 'Columns', 'ContentDate', 'ContentTime', 'PatientSex',
         'PatientBirthDate', 'ReferringPhysicianName', 'Rows', 'SOPClassUID', 'SOPInstanceUID', 'StudyDate',
         'StudyID', 'StudyInstanceUID', 'StudyTime', 'InstanceNumber', 'SeriesInstanceUID', 'SeriesNumber'],
        axis=1, inplace=True)
    print(dicom_info_clean.info())

    # dicom cleaned holds nan values in the columns: 'Laterality' and 'SeriesDescription'
    dicom_info_clean["Laterality"].fillna(method="backfill", axis=0, inplace=True)
    dicom_info_clean["SeriesDescription"].fillna(method="backfill", axis=0, inplace=True)
    print(dicom_info_clean.info())

    """
    Calcification data cleaning
    """
    calc_case_clean = df_calc_case.copy()
    calc_case_clean = calc_case_clean.rename(columns={'calc type': 'calc_type'})
    calc_case_clean = calc_case_clean.rename(columns={'calc distribution': 'calc_distribution'})
    calc_case_clean = calc_case_clean.rename(columns={'image view': 'image_view'})
    calc_case_clean = calc_case_clean.rename(columns={'left or right breast': 'left_or_right_breast'})
    calc_case_clean = calc_case_clean.rename(columns={'breast density': 'breast_density'})
    calc_case_clean = calc_case_clean.rename(columns={'abnormality type': 'abnormality_type'})
    calc_case_clean['pathology'] = calc_case_clean['pathology'].astype('category')
    calc_case_clean['calc_type'] = calc_case_clean['calc_type'].astype('category')
    calc_case_clean['calc_distribution'] = calc_case_clean['calc_distribution'].astype('category')
    calc_case_clean['abnormality_type'] = calc_case_clean['abnormality_type'].astype('category')
    calc_case_clean['image_view'] = calc_case_clean['image_view'].astype('category')
    calc_case_clean['left_or_right_breast'] = calc_case_clean['left_or_right_breast'].astype('category')
    calc_case_clean.isna().sum()

    calc_case_clean['calc_type'].fillna(method='bfill', axis=0, inplace=True)
    calc_case_clean['calc_distribution'].fillna(method='bfill', axis=0, inplace=True)
    calc_case_clean.isna().sum()

    """
    Mass data cleaning
    """
    mass_case_clean = df_mass_case.copy()
    mass_case_clean = mass_case_clean.rename(columns={'mass shape': 'mass_shape'})
    mass_case_clean = mass_case_clean.rename(columns={'left or right breast': 'left_or_right_breast'})
    mass_case_clean = mass_case_clean.rename(columns={'mass margins': 'mass_margins'})
    mass_case_clean = mass_case_clean.rename(columns={'image view': 'image_view'})
    mass_case_clean = mass_case_clean.rename(columns={'abnormality type': 'abnormality_type'})
    mass_case_clean['left_or_right_breast'] = mass_case_clean['left_or_right_breast'].astype('category')
    mass_case_clean['image_view'] = mass_case_clean['image_view'].astype('category')
    mass_case_clean['mass_margins'] = mass_case_clean['mass_margins'].astype('category')
    mass_case_clean['mass_shape'] = mass_case_clean['mass_shape'].astype('category')
    mass_case_clean['abnormality_type'] = mass_case_clean['abnormality_type'].astype('category')
    mass_case_clean['pathology'] = mass_case_clean['pathology'].astype('category')
    mass_case_clean.isna().sum()

    mass_case_clean['mass_shape'].fillna(method='bfill', axis=0, inplace=True)
    mass_case_clean['mass_margins'].fillna(method='bfill', axis=0, inplace=True)
    mass_case_clean.isna().sum()

    print("Stop here")
