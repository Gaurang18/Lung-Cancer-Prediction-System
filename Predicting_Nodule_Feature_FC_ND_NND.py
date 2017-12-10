#import MODEL_3D.Nodule_3DCNN as FC
import Nodule_Feature_FC_ND_NND as FC
import csv
import numpy
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
import matplotlib.pyplot as plt
import keras
import csv
import os
import cv2
from skimage.measure import label,regionprops
from sklearn.decomposition import PCA as sklearnPCA
#from keras.utils import plot_model
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.models import Model
import csv

DATA_FOLDER = '/home/kshitij/Desktop/Dataset/Features/KAGGLE_NODULE_CROP_ALL_NPY_VAL32/'
CSV_KAGGLE_PATH = '/home/kshitij/Desktop/Dataset/stage2_sample_submission.csv'
#CSV_PATH = '/root/workspace/dsb2017_tutorial/3d_conv/lidc/csv/annotations_nodule_label_aug.csv'
CSV_LIDC_TRAINING_PATH = 'annotations_luna_training.csv'
CSV_LIDC_TEST_PATH = 'annotations_luna_test.csv'
#NUMPY_ROOT_PATH = '/root/workspace/dsb2017_tutorial/3d_conv/lidc/classification_crop_all/'
NUMPY_ROOT_PATH = './classification_rect2/'
ORI_PATH = "rescale/"
FLIP_HORI_PATH = "rescale_flip/hori/"
FLIP_VERT_PATH = "rescale_flip/vert/"

R_ORI_PATH = "rescale_rotate/ori/"
R_FLIP_HORI_PATH = "rescale_rotate/flip_hori/"
R_FLIP_VERT_PATH = "rescale_rotate/flip_vert/"

IMAGE_FEATURE_PATH = './'

AUG_PATH_LIST = [ORI_PATH,FLIP_HORI_PATH,FLIP_VERT_PATH,R_ORI_PATH,R_FLIP_HORI_PATH,R_FLIP_VERT_PATH]
#np.set_printoptions(threshold=np.nan)

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.test_losses = []
        self.val_losses = []
        self.val_acc = []
        # self.f_losses = []
        # self.f_val_losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.test_losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))
        # self.f_losses.append(logs.get('feature_loss'))
        # self.f_val_losses.append(logs.get('val_feature_loss'))
        plt.plot(self.test_losses)
        plt.plot(self.val_losses)
        plt.plot(self.val_acc)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig("./Nodule_Feature_FC_ND_NND.png")
        plt.close()

        # plt.plot(self.f_losses)
        # plt.plot(self.f_val_losses)
        # plt.title('model loss')
        # plt.ylabel('loss')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'test'], loc='upper left')
        # plt.savefig("./Nodule_2DCNN_DUAL_xception_multi_label_feature.png")
        # plt.close()

def Kaggle_Data_Load(DATA_FOLDER, CSV_KAGGLE_PATH, Training_Sample , Test_Sample):
    patients = []

    with open(CSV_KAGGLE_PATH) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            patients.append([row['id'], row['cancer']])

    # patients = os.listdir(INPUT_FOLDER)
    #patients.sort()
    patients.sort(key=lambda x: str(x[0]), reverse=False)
    #print patients

    NonNodule_filelist = []
    #NonNodule_label = []

    for i in xrange(505):
	print i
        if patients[i][1] == str(0):
            nodules = os.listdir(DATA_FOLDER+"V"+str(i)+"/")
            for j in xrange(len(nodules)):
                #print nodules[j]+ "V" + str(i) + "_" + patients[i][0] + "_ND" + str(j) + ".npy"
                NonNodule_filelist.append([DATA_FOLDER+"V"+str(i)+"/"+nodules[j],0])
                #NonNodule_label.append(0)
    #print numpy.shape(NonNodule_filelist)
    #print (NonNodule_filelist[0])


    NonNodule_traing_filelist = NonNodule_filelist[0:Training_Sample]
    NonNodule_test_filelist = NonNodule_filelist[Training_Sample+1:min(Training_Sample+Test_Sample+1, len(NonNodule_filelist)-1)]

    NonNodule_traing_filelist = shuffle(NonNodule_traing_filelist)
    NonNodule_test_filelist = shuffle(NonNodule_test_filelist)

    return NonNodule_traing_filelist, NonNodule_test_filelist

def LIDC_Data_Load( NUMPY_ROOT_PATH,CSV_LIDC_TRAINING_PATH, CSV_LIDC_TEST_PATH ,ORI_PATH, FLIP_HORI_PATH, FLIP_VERT_PATH, AUG_PATH_LISTl ):

    Nodule_training_filelist = []
    Nodule_test_filelist = []

    with open(CSV_LIDC_TRAINING_PATH) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            filename = NUMPY_ROOT_PATH+AUG_PATH_LIST[int(row['aug'])-1]+row['pt_id']+"&"+row['n_id']+".npy"
            data = [filename, 1]
            Nodule_training_filelist.append(data)
    #print Nodule_training_filelist[0]

    #print numpy.shape(Nodule_training_filelist)
    Nodule_training_filelist = shuffle(Nodule_training_filelist)

    with open(CSV_LIDC_TEST_PATH) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            filename = NUMPY_ROOT_PATH+AUG_PATH_LIST[int(row['aug'])-1]+row['pt_id']+"&"+row['n_id']+".npy"
            data = [filename, 1]
            Nodule_test_filelist.append(data)
    #print Nodule_test_filelist[0]

    #print numpy.shape(Nodule_test_filelist)
    Nodule_test_filelist = shuffle(Nodule_test_filelist)

    return Nodule_training_filelist, Nodule_test_filelist


def make_diagonal_image(vol, FIX_LEN = 70):
    center = int(FIX_LEN/2)
    rotate = 45
    M = cv2.getRotationMatrix2D((center, center), rotate, 1)
    diagonal_vol = []
    for i in range(0,FIX_LEN):
        diagonal_vol.append(cv2.warpAffine(vol[i], M, (FIX_LEN, FIX_LEN), borderValue=-2000))

    diagonal_vol = numpy.array(diagonal_vol, dtype=numpy.float32)
    return numpy.transpose(diagonal_vol,(1, 2, 0))[35], numpy.transpose(diagonal_vol,(2,1,0))[35]

def Global_Thresholding_3D(vol):
    mask_vol = numpy.zeros((numpy.shape(vol)[0], numpy.shape(vol)[1], numpy.shape(vol)[2]), dtype = int)
    temp_vol = numpy.copy(vol)
    #temp_vol[vol>-500] = -2000
    box_array = []
    for i in xrange(0, numpy.shape(vol)[0],4):
        for j in xrange(0, numpy.shape(vol)[1],4):
            for k in xrange(0, numpy.shape(vol)[2],4):
                if temp_vol[i,j,k]>-1000 and temp_vol[i,j,k]<100:
                    box_array.append(temp_vol[i,j,k])

    box_array_np = numpy.asarray(box_array)
    # print "boxarray", np.shape(box_array_np)
    # print "mean", box_array_np.mean(),box_array_np.max()
    #print np.amin(box_array_np),np.amax(box_array_np), np.mean(box_array_np)
    #mask_vol[vol>(box_array_np.mean() + 0.0* box_array_np.std())] = True
    mask_vol[vol > (0.8*box_array_np.mean() + 0.2*box_array_np.max()) ] = 1
    mask_vol[vol > 100] = 0



    return mask_vol

def ND_Analysis_2D(nodule_Img, mask):
    ND_Score = 0
    FeatureVec = []
    if numpy.amax(mask) >0:
        FeatureVec.append(regionprops(mask,nodule_Img)[0].minor_axis_length/(regionprops(mask)[0].major_axis_length+0.0000000001))
        #FeatureVec.append(regionprops(mask,nodule_Img)[0].area)
        #FeatureVec.append(regionprops(mask,nodule_Img)[0].convex_area)
        FeatureVec.append(regionprops(mask,nodule_Img)[0].eccentricity)
        #FeatureVec.append(regionprops(mask,nodule_Img)[0].equivalent_diameter)
        FeatureVec.append(regionprops(mask,nodule_Img)[0].extent)
        #FeatureVec.append(regionprops(mask,nodule_Img)[0].filled_area)
        FeatureVec.append(regionprops(mask,nodule_Img)[0].inertia_tensor[0][0])
        FeatureVec.append(regionprops(mask,nodule_Img)[0].inertia_tensor[0][1])
        FeatureVec.append(regionprops(mask,nodule_Img)[0].inertia_tensor[1][0])
        FeatureVec.append(regionprops(mask,nodule_Img)[0].inertia_tensor[1][1])
        # FeatureVec.append(regionprops(mask,nodule_Img)[0].inertia_tensor_eigvals[0])
        # FeatureVec.append(regionprops(mask,nodule_Img)[0].inertia_tensor_eigvals[1])
        FeatureVec.append(regionprops(mask, nodule_Img)[0].inertia_tensor_eigvals[1] / (regionprops(mask)[0].inertia_tensor_eigvals[0] + 0.0000000001))
        # FeatureVec.append(regionprops(mask,nodule_Img)[0].major_axis_length)
        # FeatureVec.append(regionprops(mask,nodule_Img)[0].minor_axis_length)
        FeatureVec.append(regionprops(mask,nodule_Img)[0].max_intensity)
        FeatureVec.append(regionprops(mask,nodule_Img)[0].mean_intensity)
        FeatureVec.append(regionprops(mask,nodule_Img)[0].min_intensity)
        FeatureVec.append(regionprops(mask,nodule_Img)[0].moments[0][0])
        FeatureVec.append(regionprops(mask,nodule_Img)[0].moments[0][1])
        FeatureVec.append(regionprops(mask,nodule_Img)[0].moments[0][2])
        FeatureVec.append(regionprops(mask,nodule_Img)[0].moments[1][0])
        FeatureVec.append(regionprops(mask,nodule_Img)[0].moments[1][1])
        FeatureVec.append(regionprops(mask,nodule_Img)[0].moments[1][2])
        FeatureVec.append(regionprops(mask,nodule_Img)[0].moments[2][0])
        FeatureVec.append(regionprops(mask,nodule_Img)[0].moments[2][1])
        FeatureVec.append(regionprops(mask,nodule_Img)[0].moments[2][2])
        FeatureVec.append(regionprops(mask,nodule_Img)[0].moments_hu[0])
        FeatureVec.append(regionprops(mask,nodule_Img)[0].moments_hu[1])
        FeatureVec.append(regionprops(mask, nodule_Img)[0].orientation)
        FeatureVec.append(regionprops(mask, nodule_Img)[0].perimeter)
        FeatureVec.append(regionprops(mask, nodule_Img)[0].solidity)


        #print FeatureVec


    else:
        FeatureVec = numpy.zeros(25)

    return FeatureVec

def check_range(coord_X, coord_Y, coord_Z):
    x_max = numpy.amax(coord_X)
    x_min = numpy.amin(coord_X)
    y_max = numpy.amax(coord_Y)
    y_min = numpy.amin(coord_Y)
    z_max = numpy.amax(coord_Z)
    z_min = numpy.amin(coord_Z)

    ##real Bounding Box length
    x_rng = x_max - x_min
    y_rng = y_max - y_min
    z_rng = z_max - z_min

    max_rng = max(max(x_rng, y_rng), z_rng)

    Radius = max_rng/2

    return Radius

def PCA_3D(mask):

    region = regionprops(mask)[0]
    points = []
    coord_X = []
    coord_Y = []
    coord_Z = []
    idx = 0
    for coordinates in region.coords:
        points.append([coordinates[0], coordinates[1], coordinates[2]])
        coord_X.append(coordinates[2])
        coord_Y.append(coordinates[1])
        coord_Z.append(coordinates[0])
        # Input_vol[coord_Z, coord_Y, coord_X] = 255

    data = numpy.array(points)
    coord_mean = []
    coord_mean.append(numpy.mean(coord_Z))
    coord_mean.append(numpy.mean(coord_Y))
    coord_mean.append(numpy.mean(coord_X))


    R  = check_range(coord_X, coord_Y, coord_Z)

    sklearn_pca = sklearnPCA()
    sklearn_pca.fit(data)
    lamda1 = sklearn_pca.explained_variance_[0] / (
        sklearn_pca.explained_variance_[0] + sklearn_pca.explained_variance_[1] + sklearn_pca.explained_variance_[
            2])
    lamda2 = sklearn_pca.explained_variance_[1] / (
        sklearn_pca.explained_variance_[0] + sklearn_pca.explained_variance_[1] + sklearn_pca.explained_variance_[
            2])
    lamda3 = sklearn_pca.explained_variance_[2] / (
        sklearn_pca.explained_variance_[0] + sklearn_pca.explained_variance_[1] + sklearn_pca.explained_variance_[
            2])

    ELScore = lamda3/(lamda1+0.0000000001)
    CompScore = region.area/((4*numpy.pi*R*R*R)/3)

    return lamda1,lamda2,lamda3, ELScore, CompScore

def Intensity_based_Features_3D(ND, mask_vol):

    Idx_In = numpy.where(mask_vol == 1)
    Idx_Out = numpy.where(mask_vol == 0)

    In_arr = []
    Out_arr = []
    #print np.shape(Idx_In)
    for i in range(numpy.shape(Idx_In)[1]):
        In_arr.append(ND[Idx_In[2][i],Idx_In[1][i],Idx_In[0][i]])
    for j in range(numpy.shape(Idx_Out)[1]):
        Out_arr.append(ND[Idx_Out[2][j], Idx_Out[1][j], Idx_Out[0][j]])

    InMean = numpy.mean(In_arr)
    OutMean = numpy.mean(Out_arr)
    InVar = numpy.std(In_arr)*numpy.std(In_arr)
    Kurto = numpy.mean((In_arr - InMean)**4)/((InVar)**2)
    Skew = numpy.sum((In_arr-InMean)**3)/((numpy.shape(Idx_In)[1]-1)**3)


    return InMean, OutMean, InVar, Skew, Kurto

def ND_Analysis_3D(nodule_Img, mask):

    FeatureVec = []
    if numpy.amax(mask) >0:

        FeatureVec.append(regionprops(mask,nodule_Img)[0].extent)
        FeatureVec.append(regionprops(mask,nodule_Img)[0].max_intensity)
        FeatureVec.append(regionprops(mask,nodule_Img)[0].mean_intensity)
        FeatureVec.append(regionprops(mask,nodule_Img)[0].min_intensity)

        lamda1, lamda2, lamda3, ELScore, CompScore = PCA_3D(mask)

        FeatureVec.append(lamda1)
        FeatureVec.append(lamda2)
        FeatureVec.append(lamda3)
        FeatureVec.append(ELScore)
        FeatureVec.append(CompScore)

        InMean, OutMean, InVar, Skew, Kurto = Intensity_based_Features_3D(nodule_Img, mask)

        FeatureVec.append(InMean)
        FeatureVec.append(OutMean)
        FeatureVec.append(InVar)
        FeatureVec.append(Skew)
        FeatureVec.append(Kurto)

        #print InMean, OutMean, InVar, Skew, Kurto

        #print FeatureVec


    else:
        FeatureVec = numpy.zeros(14)

    return FeatureVec

def Extract_ND_Feature(ND):

    mask_vol = Global_Thresholding_3D(ND)

    nodule_Img1 = ND[int(round(numpy.shape(ND)[0] / 2)), :, :]
    nodule_Img2 = ND[:, int(round(numpy.shape(ND)[0] / 2)), :]
    nodule_Img3 = ND[:, :, int(round(numpy.shape(ND)[0] / 2))]

    mask_Img1 = mask_vol[int(round(numpy.shape(ND)[0] / 2)), :, :]
    mask_Img2 = mask_vol[:, int(round(numpy.shape(ND)[0] / 2)), :]
    mask_Img3 = mask_vol[:, :, int(round(numpy.shape(ND)[0] / 2))]

    nodule_Img4, nodule_Img5 = make_diagonal_image(ND)
    nodule_Img6, nodule_Img7 = make_diagonal_image(numpy.transpose(ND, (1, 2, 0)))
    nodule_Img8, nodule_Img9 = make_diagonal_image(numpy.transpose(ND, (2, 1, 0)))

    temp_mask_vol = numpy.array(mask_vol, dtype=numpy.uint8)
    mask_Img4, mask_Img5 = make_diagonal_image(temp_mask_vol)
    mask_Img6, mask_Img7 = make_diagonal_image(numpy.transpose(temp_mask_vol, (1, 2, 0)))
    mask_Img8, mask_Img9 = make_diagonal_image(numpy.transpose(temp_mask_vol, (2, 1, 0)))

    mask_Img4 = numpy.asarray(mask_Img4, dtype=numpy.uint8, order='C')
    mask_Img5 = numpy.asarray(mask_Img5, dtype=numpy.uint8, order='C')
    mask_Img6 = numpy.asarray(mask_Img6, dtype=numpy.uint8, order='C')
    mask_Img7 = numpy.asarray(mask_Img7, dtype=numpy.uint8, order='C')
    mask_Img8 = numpy.asarray(mask_Img8, dtype=numpy.uint8, order='C')
    mask_Img9 = numpy.asarray(mask_Img9, dtype=numpy.uint8, order='C')

    FeatureVec1 = ND_Analysis_2D(nodule_Img1, mask_Img1)
    FeatureVec2 = ND_Analysis_2D(nodule_Img2, mask_Img2)
    FeatureVec3 = ND_Analysis_2D(nodule_Img3, mask_Img3)
    FeatureVec4 = ND_Analysis_2D(nodule_Img4, mask_Img4)
    FeatureVec5 = ND_Analysis_2D(nodule_Img5, mask_Img5)
    FeatureVec6 = ND_Analysis_2D(nodule_Img6, mask_Img6)
    FeatureVec7 = ND_Analysis_2D(nodule_Img7, mask_Img7)
    FeatureVec8 = ND_Analysis_2D(nodule_Img8, mask_Img8)
    FeatureVec9 = ND_Analysis_2D(nodule_Img9, mask_Img9)
    FeatureVec10 = ND_Analysis_3D(ND, mask_vol)
    #print (np.shape(FeatureVec1))
    #print (np.shape(FeatureVec2))
    #print (np.shape(FeatureVec3))
    #print (np.shape(FeatureVec4))


    #ND_Feature = FeatureVec1 + FeatureVec2 + FeatureVec3 + FeatureVec4
    ND_Feature = numpy.concatenate((FeatureVec1, FeatureVec2, FeatureVec3, FeatureVec4, FeatureVec5,
                                 FeatureVec6, FeatureVec7, FeatureVec8, FeatureVec9, FeatureVec10))
    #print ND_Feature

    return ND_Feature

def make_image_feature(data_list):

    data_list_len = len(data_list)

    max_image_feature = numpy.load(IMAGE_FEATURE_PATH + "max_image_feature.npy")
    min_image_feature = numpy.load(IMAGE_FEATURE_PATH + "min_image_feature.npy")

    feature_label = []

    count = 0
    for data in data_list:

        count = count + 1
        print count
        file_path = data[0]
        image = numpy.load(file_path)
        image_feature = Extract_ND_Feature(image)


        for i in range(0, len(image_feature)):
            image_feature[i] = float(image_feature[i] - min_image_feature[i])\
                               / float(max_image_feature[i] - min_image_feature[i])

        if data[1] == 1:
            feature_label.append(numpy.concatenate([numpy.array(image_feature), numpy.array([1, 0])]))
        else:
            feature_label.append(numpy.concatenate([numpy.array(image_feature), numpy.array([0, 1])]))


        #feature_label.append(image_feature)

    feature_label = numpy.array(feature_label)



    return feature_label



def generate_batch(data_list,  batch_size ):

    data_list_len = len(data_list)
    #print data_list_len
    count = 0
    # max_image_feature = numpy.load(IMAGE_FEATURE_PATH + "max_image_feature.npy")
    # min_image_feature = numpy.load(IMAGE_FEATURE_PATH + "min_image_feature.npy")
    while True:
        batch_train_feature = []
        batch_train_label = []
        #for count in range(0, loop_num):
        if (count+1)*batch_size <= data_list_len:
            max_len = (count+1)*batch_size
        else:
            max_len = data_list_len
        batch_data = data_list[count*batch_size:max_len]
        for data in batch_data:
            # if data[1] == 1:
            #     batch_train_label.append([1, 0])
            # else:
            #     batch_train_label.append([0, 1])
            batch_train_label.append([data[239], data[240]])

            # file_path = data[0]
            # train_image = numpy.load(file_path)
            image_feature = data[0:238]


            # for i in range(0, len(image_feature)):
            #     image_feature[i] = float(image_feature[i] - min_image_feature[i])\
            #                        / float(max_image_feature[i] - min_image_feature[i])

            batch_train_feature.append(image_feature)

        batch_train_feature = numpy.array(batch_train_feature)
        batch_train_label = numpy.array(batch_train_label)

        count = count + 1
        if count*batch_size >= data_list_len:
            count = 0
            data_list = shuffle(data_list)

        return batch_train_feature, batch_train_label

if __name__ == '__main__':

    NonNodule_traing_filelist, NonNodule_test_filelist = Kaggle_Data_Load(DATA_FOLDER, CSV_KAGGLE_PATH, Training_Sample = 10, Test_Sample = 10)
    Training_filelist = NonNodule_traing_filelist
    Test_filelist = NonNodule_test_filelist
    Train_feature_label = make_image_feature(Training_filelist)
    Test_feature_label = make_image_feature(Test_filelist)

    # nodule_model = FC.Nodule_FC()
    # nodule_model.model.summary()
    # history = LossHistory()
    # pl = keras.callbacks.ProgbarLogger(count_mode='steps')
    # #data_lists = data_lists[:100]
    #   
    # if os.path.isfile('./my_modeller.hdf5'):
    #    print ("load ./my_modeller.hdf5")
    #    nodule_model.model = load_model('./my_modeller.hdf5')
    
    # callbacks = [
    #     EarlyStopping(monitor='val_loss', patience=10000, verbose=1),
    #     history,pl,
    #     #ModelCheckpoint('my_modellerllhh.hdf5', monitor='val_loss', save_best_only=True, verbose=1),
    # ]

    #print Test_feature_label.shape
    at = [Train_feature_label[i:i+24] for i in range(0, len(Train_feature_label), 24)]
    bt = [Test_feature_label[i:i+24] for i in range(0, len(Test_feature_label), 24)]
    # batch_size = 10
    # epoch = 4
    # nb_val_samples = len(Test_filelist)
    # cx,bx = generate_batch(Test_feature_label, batch_size)
    # writer = csv.writer(file,delimiter=',')
    #             writer.writerow(result_list)

    with open("BatchTrain.csv", 'wb') as myfile:
        wr = csv.writer(myfile,delimiter=',')
        for i in range(0, len(at)):
            for j in range(0, len(at[i])):
                wr.writerow(at[i][j])
        # for sublist in at:
        #     #for item in sublist:
        #     wr.writerow(sublist)
        #     #wr.write('\n')
    # predictions = nodule_model.model.predict(cx)
    # for i in range(predictions.size/2):
    #     print predictions[i][0]

    # a  = predictions
    # np.savetxt('np.csv', a, fmt='%.5f', delimiter=',', header=" #1,  #2")
