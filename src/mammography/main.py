import pandas as pd
import evaluation
from process import get_images
# from DCUNet import DCUNet
import efficientNet
from evaluation import tversky, tversky_loss, focal_tversky, dice_coef, dice_coef_loss, jacard
import numpy as np
from sklearn.model_selection import train_test_split
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


if __name__ == '__main__':
    script_dir = os.path.dirname(__file__)
    data_path = os.path.join(script_dir + "/../data")

    dicom_info = pd.read_csv(data_path + "/CBIS-DDSM/dicom_info.csv")
    image_dir = data_path + "/CBIS-DDSM/jpeg"

    print(dicom_info.head())
    print(dicom_info.info())

    full_mammogram_images = get_images(img_type="full mammogram images", image_dir=image_dir,
                                       show_img=False, dicom_info=dicom_info)
    ROI_images = get_images(img_type="ROI mask images", image_dir=image_dir,
                                       show_img=False, dicom_info=dicom_info)

    X = np.array((2000, 249, 249))
    Y = np.array((2000, 249, 249))
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5)

    # dcunet = DCUNet(height=250, width=250, channels=1)
    # dcunet.compile(optimizer='adam', loss=focal_tversky, metrics=[dice_coef, jacard, 'accuracy'])
    # dcunet.summary()

    effnetV2 = efficientNet.effnetv2_s()
    effnetV2.summary()

    evaluation.trainStep(effnetV2, X_train, Y_train, X_test, Y_test, epochs=150, batchSize=4)

    print("Stop here")
