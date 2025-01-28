import pandas as pd
import matplotlib.pyplot as plt
import PIL
from model import DCUNet
from evaluation import tversky, tversky_loss, focal_tversky, dice_coef, dice_coef_loss, jacard
import numpy as np
from sklearn.model_selection import train_test_split
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

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

Kaggle Dataset: https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset?resource=download
Kaggle Notebook: https://www.kaggle.com/code/baselanaya/breast-cancer-detection-using-cnn/notebook
DC-UNet: https://github.com/AngeLouCN/DC-UNet/blob/main/main.py
"""


if __name__ == '__main__':
    script_dir = os.path.dirname(__file__)
    data_path = os.path.join(script_dir + "/../data")

    dicom_info = pd.read_csv(data_path + "/dicom_info.csv")
    image_dir = data_path + "/jpeg"

    print(dicom_info.head())
    print(dicom_info.info())

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
    Training and Evaluation
    """
    def saveModel(model):
        model_json = model.to_json()
        try:
            os.makedirs('models')
        except:
            pass
        fp = open('models/modelP.json', 'w')
        fp.write(model_json)
        model.save('models/modelW.h5')

    def evaluateModel(model, X_test, Y_test, batchSize):
        try:
            os.makedirs('results')
        except:
            pass
        yp = model.predict(x=X_test, batch_size=batchSize, verbose=1)
        yp = np.round(yp, 0)
        for i in range(10):
            plt.figure(figsize=(20, 10))
            plt.subplot(1, 3, 1)
            plt.imshow(X_test[i])
            plt.title('Input')
            plt.subplot(1, 3, 2)
            plt.imshow(Y_test[i].reshape(Y_test[i].shape[0], Y_test[i].shape[1]))
            plt.title('Ground Truth')
            plt.subplot(1, 3, 3)
            plt.imshow(yp[i].reshape(yp[i].shape[0], yp[i].shape[1]))
            plt.title('Prediction')

            intersection = yp[i].ravel() * Y_test[i].ravel()
            union = yp[i].ravel() + Y_test[i].ravel() - intersection

            jacard = (np.sum(intersection) / np.sum(union))
            plt.suptitle('Jacard Index' + str(np.sum(intersection)) + '/' + str(np.sum(union)) + '=' + str(jacard))

            plt.savefig('results/' + str(i) + '.png', format='png')
            plt.close()

        jacard = 0
        dice = 0
        for i in range(len(Y_test)):
            yp_2 = yp[i].ravel()
            y2 = Y_test[i].ravel()

            intersection = yp_2 * y2
            union = yp_2 + y2 - intersection

            jacard += (np.sum(intersection) / np.sum(union))

            dice += (2. * np.sum(intersection)) / (np.sum(yp_2) + np.sum(y2))

        jacard /= len(Y_test)
        dice /= len(Y_test)

        print('Jacard Index : ' + str(jacard))
        print('Dice Coefficient : ' + str(dice))

        fp = open('models/log.txt', 'a')
        fp.write(str(jacard) + '\n')
        fp.close()

        fp = open('models/best.txt', 'r')
        best = fp.read()
        fp.close()

        if (jacard > float(best)):
            print('***********************************************')
            print('Jacard Index improved from ' + str(best) + ' to ' + str(jacard))
            print('***********************************************')
            fp = open('models/best.txt', 'w')
            fp.write(str(jacard))
            fp.close()

            saveModel(model)

    def trainStep(model, X_train, Y_train, X_test, Y_test, epochs, batchSize):

        for epoch in range(epochs):
            print('Epoch : {}'.format(epoch + 1))
            model.fit(x=X_train, y=Y_train, batch_size=batchSize, epochs=1, verbose=1)

            evaluateModel(model, X_test, Y_test, batchSize)

        return model

    X = np.array((2000, 250, 250))
    Y = np.array((2000, 250, 250))
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5)

    model = DCUNet(height=250, width=250, channels=1)
    model.compile(optimizer='adam', loss=focal_tversky, metrics=[dice_coef, jacard, 'accuracy'])
    model.summary()

    trainStep(model, X_train, Y_train, X_test, Y_test, epochs=150, batchSize=4)

    print("Stop here")
