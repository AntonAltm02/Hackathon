import pandas as pd
import matplotlib.pyplot as plt
from process import get_images
from DCUNet import DCUNet
from evaluation import tversky, tversky_loss, focal_tversky, dice_coef, dice_coef_loss, jacard
import numpy as np
from sklearn.model_selection import train_test_split
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


if __name__ == '__main__':
    script_dir = os.path.dirname(__file__)
    data_path = os.path.join(script_dir + "/../data")

    dicom_info = pd.read_csv(data_path + "/dicom_info.csv")
    image_dir = data_path + "/jpeg"

    print(dicom_info.head())
    print(dicom_info.info())

    full_mammogram_images = get_images(img_type="full mammogram images", image_dir=image_dir,
                                       show_img=False, dicom_info=dicom_info)
    ROI_images = get_images(img_type="ROI mask images", image_dir=image_dir,
                                       show_img=False, dicom_info=dicom_info)

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

    X = np.array((2000, 249, 249))
    Y = np.array((2000, 249, 249))
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5)

    model = DCUNet(height=250, width=250, channels=1)
    model.compile(optimizer='adam', loss=focal_tversky, metrics=[dice_coef, jacard, 'accuracy'])
    model.summary()

    trainStep(model, X_train, Y_train, X_test, Y_test, epochs=150, batchSize=4)

    print("Stop here")
