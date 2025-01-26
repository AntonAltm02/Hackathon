import os
import pandas as pd
import matplotlib.pyplot as plt
import PIL

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

    print("Stop here")
