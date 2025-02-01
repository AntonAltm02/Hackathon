import matplotlib.pyplot as plt
import PIL

def calcification_data_cleaning(df_calc_case):
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

def mass_data_cleaning(df_mass_case):
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

def get_images(img_type, show_img, image_dir, dicom_info):
    """
    This function gets all the images from the dicom folder based on the specified image type
    :param show_img: boolean parameter whether the images should be shown or not
    :param img_type:  might be set to "cropped images", "full mammogram images" or "ROI mask images"
    :return:
    """
    images = dicom_info[dicom_info.SeriesDescription == img_type].image_path
    images.head()
    images = images.apply(lambda x: x.replace('CBIS-DDSM/jpeg', image_dir))
    images.head()
    for file in images[0:10]:
        images_show = PIL.Image.open(file)
        gray_img = images_show.convert("L")
        plt.imshow(gray_img, cmap='gray')
        plt.title(img_type)
        if show_img:
            plt.show()
    return images