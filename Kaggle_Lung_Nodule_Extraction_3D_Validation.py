# %matplotlib inline
import time
from skimage.morphology import disk, binary_erosion, binary_closing, binary_opening
from skimage.measure import label,regionprops
from skimage.segmentation import clear_border
import scipy.misc
import scipy.ndimage
import scipy.misc
import csv
import pylidc as pl
from scipy.stats import threshold
import numpy as np # linear algebra
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import copy

from skimage import measure

from skimage.morphology import convex_hull_image
from skimage.feature import peak_local_max
from skimage.morphology import watershed, disk
from skimage.segmentation import random_walker

try:
    from skimage import filters
except ImportError:
    from skimage import filter as filters

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sklearn.decomposition import PCA as sklearnPCA
from skimage.filters import roberts, sobel
from scipy import ndimage as ndi
from subprocess import check_output
import dicom
import os

import cv2
from skimage.morphology import square

INPUT_FOLDER = '/home/kshitij/Desktop/Dataset/stage2/'
OUTPUT_FOLDER1 = '/home/kshitij/Desktop/Dataset/KAGGLE_NODULE_CROP_ALL_XY2/'
OUTPUT_FOLDER2 = '/home/kshitij/Desktop/Dataset//KAGGLE_NODULE_CROP_ALL_YZ2/'
OUTPUT_FOLDER3 = '/home/kshitij/Desktop/Dataset/KAGGLE_NODULE_CROP_ALL_ZX2/'
OUTPUT_FOLDER4 = '/home/kshitij/Desktop/Dataset/KAGGLE_NODULE_CROP_ALL_NPY_VAL32/'

# Load the scans in given folder path
def load_scan(path):

    fnames = []
    for filename in os.listdir(path):
        if filename.endswith('.dcm'):
            try:
                fnames.append(filename)
            except dicom.errors.InvalidDicomError as exc:
                print ("something wrong with", path + '/' + filename)

    slices = []
    for dicom_file_name in fnames:
        with open(os.path.join(path, dicom_file_name), 'rb') as f:
            slices.append( dicom.read_file(f) )

    #slices.sort(key=lambda x: int(x.InstanceNumber))
    slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices

def load_scans():
    qu = pl.query(pl.Scan)
    # qu = pl.query(pl.Scan).filter(pl.Scan.patient_id == "LIDC-IDRI-0010")
    # print qu.count()
    scans = qu.all()

    def getKey(item):
        return item.patient_id

    scans = sorted(scans, key=getKey)

    # from patient_id: 0510
    # scans = scans[516::]

    return scans


def load_annotations(scan):
    anns = scan.annotations
    return anns


def attributes_to_csv(ann, scan, z):
    # with open('annotations.csv', 'aw') as csvfile:
    with open('/home/kshitij/Desktop/Dataset/annotations.csv', 'aw') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        try:
            writebuf = [ann.scan.patient_id, z, scan.slice_thickness, scan.pixel_spacing,
                        ann.Subtlety(), ann.InternalStructure(), ann.Calcification(), ann.Sphericity(),
                        ann.Margin(), ann.Lobulation(), ann.Spiculation(), ann.Texture(), ann.Malignancy(),
                        ann.estimate_diameter(), ann.estimate_volume()]
        except:
            try:
                writebuf = [ann.scan.patient_id, z, -1, scan.pixel_spacing]
            except:
                writebuf = [ann.scan.patient_id, z]
        spamwriter.writerow(writebuf)

        writer = csv.writer(csvfile)
        csvfile.close()


def save_volume(vol, fn):
    plt.imshow(vol, cmap=plt.cm.gray)
    plt.savefig(fn)
    plt.close()

def resample(image, scan, new_spacing=[1, 1, 1]):
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='constant')

    return image, new_spacing

def get_pixels_hu(images, pixel, minimum, idx):
    #image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    #image = image.astype(np.int16)

   # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0

    intercept = images[idx].RescaleIntercept
    slope = images[idx].RescaleSlope

    #for image in images:
    # if (minimum < 0):
    #     value = min(images[len(images) - 1].pixel_array[10][250], images[0].pixel_array[10][250], images[len(images) - 1].pixel_array[500][250], images[0].pixel_array[500][250])
    #     #print value
    #     pixel[pixel == minimum] = value

    if slope != 1:
        pixel = slope * pixel.astype(np.float64)
        pixel = pixel.astype(np.int16)

    pixel += np.int16(intercept)

    return pixel


def get_segmented_lungs(im, plot=False):
    '''
    This funtion segments the lungs from the given 2D slice.
    '''
    output = np.copy(im)
    if plot == True:
        f, plots = plt.subplots(8, 1, figsize=(5, 40))
    # else:
    #     im[im == -2000] = 0
    #
    #     plt.figure(1)
    #     plt.imshow(im, cmap=plt.cm.gray)
    '''
    Step 1: Convert into a binary image.
    '''


    #im = get_pixels_hu_image(im)
    #binary = im < 604
    binary = im < -400


    if plot == True:
        plots[0].axis('off')
        plots[0].imshow(binary, cmap=plt.cm.bone)
    # else:
    #
    #     plt.figure(2)
    #     plt.imshow(binary, cmap=plt.cm.gray)
    #     #plt.show()
    '''
    Step 2: Remove the blobs connected to the border of the image.
    '''
    cleared = clear_border(binary)
    Img = cleared
    #Img = 0

    Idx = np.where(cleared == 1)

    SumX = 0
    SumY = 0
    for i in range(np.shape(Idx)[1]):
        SumX = SumX + Idx[0][i]
        SumY = SumY + Idx[1][i]

    SumX = SumX / (np.shape(Idx)[1]+0.0000001)
    SumY = SumY / (np.shape(Idx)[1]+0.0000001)

    SumY = int(SumY)

    cleared[:, max(0,SumY)] = 0

    if plot == True:
        plots[1].axis('off')
        plots[1].imshow(cleared, cmap=plt.cm.bone)
    # else:
    #     plt.figure(3)
    #     plt.imshow(cleared, cmap='gray', vmin = 0, vmax = 1)
    #     plt.show()

    # selem = disk(3)
    # cleared = binary_closing(cleared, selem)
    '''
    Step 3: Label the image.
    '''
    label_image = label(cleared)
    if plot == True:
        plots[2].axis('off')
        plots[2].imshow(label_image, cmap=plt.cm.bone)
    # else:
    #     plt.figure(4)
    #     plt.imshow(label_image, cmap=plt.cm.gray)
    #     #plt.show()
    '''
    Step 4: Keep the labels with 2 largest areas.
    '''

    areas = [r.area for r in regionprops(label_image)]
    areas.sort()

    size = (w, h) = (np.shape(im)[0], np.shape(im)[1])
    binary = np.zeros(size, np.uint8)
    for region in regionprops(label_image):
        size = (w, h) = (np.shape(im)[0], np.shape(im)[1])
        each_label = np.zeros(size, np.uint8)
        if region.area>100:
            for coordinates in region.coords:
                each_label[coordinates[0], coordinates[1]] = 1
            binary = np.bitwise_or(binary,convex_hull_image(each_label))
    selem = disk(1)
    binary = binary_closing(binary, selem)
    binary = binary_erosion(binary, selem)
    # if plot == True:
    #     plots[3].axis('off')
    #     plots[3].imshow(binary, cmap=plt.cm.bone)
    # # else:
    # plt.figure(5)
    # plt.imshow(binary, cmap=plt.cm.gray)
    # plt.show()



    '''
    Step 8: Superimpose the binary mask on the input image.
    '''
    get_high_vals = binary == 0
    output[get_high_vals] = -2000
    #im[get_high_vals] = -700
    if plot == True:
        plots[8].axis('off')
        plots[8].imshow(output, cmap=plt.cm.bone)
    # else:
    #     plt.figure(10)
    #     plt.imshow(output, cmap=plt.cm.gray)
    #     #plt.show()

    return output
def get_segmented_lungs_Enhance(im, plot=False):
    '''
    This funtion segments the lungs from the given 2D slice.
    '''
    output = np.copy(im)



    # print np.amin(im)
    if plot == True:
        f, plots = plt.subplots(8, 1, figsize=(5, 40))
    # else:
    #     im[im == -2000] = 0
    #
    #     plt.figure(1)
    #     plt.imshow(im, cmap=plt.cm.gray)
    '''
    Step 1: Convert into a binary image.
    '''
    #im = get_pixels_hu_image(im)
    #binary = im < 604
    binary = im < -400

    if plot == True:
        plots[0].axis('off')
        plots[0].imshow(binary, cmap=plt.cm.bone)
    # else:
    #
    #     plt.figure(2)
    #     plt.imshow(binary, cmap=plt.cm.gray)
    #     #plt.show()
    '''
    Step 2: Remove the blobs connected to the border of the image.
    '''
    cleared = clear_border(binary)
    Img = cleared
    #Img = 0

    Idx = np.where(cleared == 1)

    SumX = 0
    SumY = 0
    for i in range(np.shape(Idx)[1]):
        SumX = SumX + Idx[0][i]
        SumY = SumY + Idx[1][i]

    SumX = SumX / (np.shape(Idx)[1]+0.0000001)
    SumY = SumY / (np.shape(Idx)[1]+0.0000001)

    SumY = int(SumY)

    cleared[:, max(0,SumY)] = 0

    if plot == True:
        plots[1].axis('off')
        plots[1].imshow(cleared, cmap=plt.cm.bone)
    # else:
    #     plt.figure(3)
    #     plt.imshow(cleared, cmap='gray', vmin = 0, vmax = 1)
    #     plt.show()

    # selem = disk(3)
    # cleared = binary_closing(cleared, selem)
    '''
    Step 3: Label the image.
    '''
    label_image = label(cleared)
    if plot == True:
        plots[2].axis('off')
        plots[2].imshow(label_image, cmap=plt.cm.bone)
    # else:
    #     plt.figure(4)
    #     plt.imshow(label_image, cmap=plt.cm.gray)
    #     #plt.show()
    '''
    Step 4: Keep the labels with 2 largest areas.
    '''

    areas = [r.area for r in regionprops(label_image)]
    areas.sort()

    size = (w, h) = (np.shape(im)[0], np.shape(im)[1])
    binary = np.zeros(size, np.uint8)
    binary_before = np.zeros(size, np.uint8)
    for region in regionprops(label_image):
        size = (w, h) = (np.shape(im)[0], np.shape(im)[1])
        each_label = np.zeros(size, np.uint8)
        if region.area>100:
            for coordinates in region.coords:
                each_label[coordinates[0], coordinates[1]] = 1
            binary = np.bitwise_or(binary,convex_hull_image(each_label))
            binary_before = np.bitwise_or(binary_before, each_label)
    selem = disk(1)
    binary = binary_closing(binary, selem)
    binary = binary_erosion(binary, selem)
    # if plot == True:
    #     plots[3].axis('off')
    #     plots[3].imshow(binary, cmap=plt.cm.bone)
    # # else:
    # plt.figure(5)
    # plt.imshow(binary, cmap=plt.cm.gray)
    # plt.show()
    '''
    Mask
    before
    Convexhull
    '''
    # plt.figure(7)
    # plt.imshow(binary_before, cmap=plt.cm.gray)
    # plt.show()
    edges = roberts(binary_before)
    binary_befour = ndi.binary_fill_holes(edges)
    if plot == True:
        plots[6].axis('off')
        plots[6].imshow(binary, cmap=plt.cm.bone)

    # plt.figure(8)
    # plt.imshow(binary_before, cmap=plt.cm.gray)
    # plt.show()


    '''
    Step 8: Superimpose the binary mask on the input image.
    '''
    get_high_vals = binary == 0
    output[get_high_vals] = -2000
    #im[get_high_vals] = -700
    if plot == True:
        plots[8].axis('off')
        plots[8].imshow(output, cmap=plt.cm.bone)
    # else:
    #     plt.figure(10)
    #     plt.imshow(output, cmap=plt.cm.gray)
    #     #plt.show()
    output = RefineLungSegmentation(output, binary_befour, binary)

    return output



def RefineLungSegmentation(im,binary_before,binary):

    binary[binary_before>0] = 0

    selem = disk(5)
    binary = binary_opening(binary, selem)
    label_image = label(binary)


    for region in regionprops(label_image):
        if region.major_axis_length>60:
            for coordinates in region.coords:
                im[coordinates[0], coordinates[1]] = -2000


    # plt.figure(13)
    # plt.imshow(im, cmap=plt.cm.gray)
    # plt.show()
    return im

def plot_2d_min(image, layer, th=-300):
    # p = image.transpose(2,1,0)
    # p = p[:,:,::-1]
    p = image.transpose(0, 1, 2)
    p = threshold(p, threshmin=th)
    print p.shape

    plt.imshow(p[layer], cmap=plt.cm.gray)
    plt.show()


def plot_2d_max(image, layer, th=-300):
    # p = image.transpose(2,1,0)
    # p = p[:,:,::-1]
    p = image.transpose(0, 1, 2)
    p = threshold(p, threshmax=th)
    print p.shape

    plt.imshow(p[layer], cmap=plt.cm.gray)
    plt.show()


def circle_filter(radius):
    kernel = np.zeros((2 * radius + 1, 2 * radius + 1))
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    mask = x ** 2 + y ** 2 <= radius ** 2
    kernel[mask] = 1
    # circular_mean = gf(data, np.mean, footprint=kernel)
    return kernel

def segment_lung_from_ct_scan(ct_scan):
    #return np.asarray([get_segmented_lungs(slice) for slice in ct_scan])
    return np.asarray([get_segmented_lungs_Enhance(slice) for slice in ct_scan])


def binary_closing_3D(vol, selem):
    return np.asarray([binary_closing(slice, selem) for slice in vol])

def binary_opening_3D(vol, selem):
    return np.asarray([binary_opening(slice, selem) for slice in vol])

#def median

def plot_ct_scan(scan):
    f, plots = plt.subplots(int(scan.shape[0] / 20) + 1, 4, figsize=(25, 25))
    print f, plots
    for i in range(0, scan.shape[0], 5):
        plots[int(i / 20), int((i % 20) / 5)].axis('off')
        plots[int(i / 20), int((i % 20) / 5)].imshow(scan[i], cmap=plt.cm.bone)


def plot_3d(image, threshold=-300):
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2, 1, 0)

    verts, faces = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')


    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()


def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

def Global_Thresholding_3D(vol):
    mask_vol = np.zeros((np.shape(vol)[0], np.shape(vol)[1], np.shape(vol)[2]), dtype=bool)
    temp_vol = np.copy(vol)
    #temp_vol[vol>-500] = -2000
    box_array = []
    for i in xrange(0, np.shape(vol)[0],4):
        for j in xrange(0, np.shape(vol)[1],4):
            for k in xrange(0, np.shape(vol)[2],4):
                if temp_vol[i,j,k]>-1000:
                    box_array.append(temp_vol[i,j,k])

    box_array_np = np.asarray(box_array)
    print "boxarray", np.shape(box_array_np)
    print "mean", box_array_np.mean(),box_array_np.max()
    #print np.amin(box_array_np),np.amax(box_array_np), np.mean(box_array_np)
    #mask_vol[vol>(box_array_np.mean() + 0.0* box_array_np.std())] = True
    mask_vol[vol > (0.8*box_array_np.mean() + 0.2*box_array_np.max()) ] = True

    return mask_vol

def Dist_Thresholding_3D(vol):
    mask_vol = np.zeros((np.shape(vol)[0], np.shape(vol)[1], np.shape(vol)[2]), dtype=bool)
    temp_vol = np.copy(vol)
    #temp_vol[vol>-500] = -2000
    box_array = []
    for i in xrange(0, np.shape(vol)[0],4):
        for j in xrange(0, np.shape(vol)[1],4):
            for k in xrange(0, np.shape(vol)[2],4):

                box_array.append(temp_vol[i,j,k])

    box_array_np = np.asarray(box_array)
    # print "boxarray", np.shape(box_array_np)
    # print "mean", box_array_np.mean(),box_array_np.max()
    #print np.amin(box_array_np),np.amax(box_array_np), np.mean(box_array_np)
    #mask_vol[vol>(box_array_np.mean() + 2.0* box_array_np.std())] = True
    mask_vol[vol > (0.8*box_array_np.mean() + 0.2*box_array_np.max()) ] = True

    return mask_vol

def Global_Thresholding_2D(img):
    mask_img = np.zeros((np.shape(img)[0], np.shape(img)[1]), dtype=int)
    temp_img = np.copy(img)

    box_array = []
    for i in xrange(0, np.shape(img)[0]):
        for j in xrange(0, np.shape(img)[1]):
                if temp_img[i,j]>-1000:
                    box_array.append(temp_img[i,j])

    box_array_np = np.asarray(box_array)
    # print "boxarray", np.shape(box_array_np)
    # print "mean", box_array_np.mean(),box_array_np.max()
    #print np.amin(box_array_np),np.amax(box_array_np), np.mean(box_array_np)
    #mask_vol[vol>(box_array_np.mean() + 0.0* box_array_np.std())] = True
    if (np.shape(box_array_np)[0] >0):
        mask_img[img > (0.8*box_array_np.mean() + 0.2*box_array_np.max()) ] = 1

    return mask_img


def hessian(x):
    """
    Calculate the hessian matrix with finite differences
    Parameters:
       - x : ndarray
    Returns:
       an array of shape (x.dim, x.ndim) + x.shape
       where the array[i, j, ...] corresponds to the second derivative x_ij
    """
    x_grad = np.gradient(x)
    hessian = np.empty((x.ndim, x.ndim) + x.shape, dtype=x.dtype)
    for k, grad_k in enumerate(x_grad):
        # iterate over dimensions
        # apply gradient again to every component of the first derivative.
        tmp_grad = np.gradient(grad_k)
        for l, grad_kl in enumerate(tmp_grad):
            hessian[k, l, :, :] = grad_kl
    return hessian


def Dot_Filtering(vol):
    hess = hessian(vol)
    mask_vol = np.zeros((np.shape(vol)[0], np.shape(vol)[1], np.shape(vol)[2]), dtype=float)

    for i in xrange(0, np.shape(vol)[0]):
        for j in xrange(0, np.shape(vol)[1]):
            for k in xrange(0, np.shape(vol)[2]):
                H = hess[:, :, i, j, k]
                W, V = np.linalg.eig(H)
                # if np.sum(W)!=0:
                #      print W
                #
                absW = abs(W)
                np.sort(absW)

                if (W[0]<0) and (W[1]<0) and (W[2]<0):
                    mask_vol[i,j,k] = ((absW[0])*(absW[0]))/(absW[2]+0.000000001)
                    #mask_vol[i, j, k] = 255
                else:
                    mask_vol[i, j, k] = 0


    return mask_vol

def nodule_filtering(vol, size_thr_up, size_thr_lw, PCA_th):
    Num_Label = 0
    label_scan = label(vol)
    areas = [r.area for r in regionprops(label_scan)]
    areas.sort()

    NoduleMask = np.zeros((np.shape(pix_resampled)[0], np.shape(pix_resampled)[1], np.shape(pix_resampled)[2]))
    NoduleMask_bool = np.zeros((np.shape(pix_resampled)[0], np.shape(pix_resampled)[1], np.shape(pix_resampled)[2]),
                               np.bool)
    for region in regionprops(label_scan):

        # if region.area < size_thr_up and region.area > size_thr_dw:
        if region.area > 5:

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

            data = np.array(points)
            # np.shape(data)

            sklearn_pca = sklearnPCA()
            sklearn_pca.fit(data)

            lamda1 = sklearn_pca.explained_variance_[0] / (
            sklearn_pca.explained_variance_[0] + sklearn_pca.explained_variance_[1] + sklearn_pca.explained_variance_[2])
            lamda2 = sklearn_pca.explained_variance_[1] / (
            sklearn_pca.explained_variance_[0] + sklearn_pca.explained_variance_[1] + sklearn_pca.explained_variance_[2])
            lamda3 = sklearn_pca.explained_variance_[2] / (
            sklearn_pca.explained_variance_[0] + sklearn_pca.explained_variance_[1] + sklearn_pca.explained_variance_[2])

            PCA_Score = abs(sklearn_pca.explained_variance_[2]) / (abs(sklearn_pca.explained_variance_[0]) + 0.00000000001)
            #PCA_Score = (3 / 2) * (lamda2 + lamda3)
            # print PCA_Score
            # print sklearn_pca.explained_variance_[0], sklearn_pca.explained_variance_[1], sklearn_pca.explained_variance_[2]
            # if pca.s[0]>1000:

            x_max, x_min, y_max, y_min, z_max, z_min = check_range(vol, coord_X, coord_Y, coord_Z, sklearn_pca.mean_, 0.2)
            X_rng = (x_max - x_min) / 1.4
            Y_rng = (y_max - y_min) / 1.4
            Z_rng = (z_max - z_min) / 1.4
            max_rng = max(X_rng, Y_rng, Z_rng)
            min_rng = min(X_rng, Y_rng, Z_rng)

            if max_rng> size_thr_lw and max_rng < size_thr_up and PCA_Score > PCA_th:

                for coordinates in region.coords:
                    NoduleMask[coordinates[0], coordinates[1], coordinates[2]] = int(255 * PCA_Score)
                    # NoduleMask[coordinates[0], coordinates[1], coordinates[2]] = 255
                    NoduleMask_bool[coordinates[0], coordinates[1], coordinates[2]] = True

                Num_Label = Num_Label + 1
                # print sklearn_pca.mean_[0], sklearn_pca.mean_[1],sklearn_pca.mean_[2]
                # plt.figure(5)
                # plt.imshow(NoduleMask[int(sklearn_pca.mean_[0])],  cmap='gray', vmin=0, vmax=255)
                # plt.figure(6)
                # plt.imshow(NoduleMask_bool[int(sklearn_pca.mean_[0])], cmap=plt.cm.gray)
                # plt.show()

    return NoduleMask, NoduleMask_bool, Num_Label


def check_range(Input_vol, coord_X, coord_Y, coord_Z,coord_mean, margin_ratio ):
        x_max = np.amax(coord_X)
        x_min = np.amin(coord_X)
        y_max = np.amax(coord_Y)
        y_min = np.amin(coord_Y)
        z_max = np.amax(coord_Z)
        z_min = np.amin(coord_Z)

        ##real Bounding Box length
        x_rng = x_max-x_min
        y_rng = y_max-y_min
        z_rng = z_max-z_min

        x_rng_r = x_max - coord_mean[2]
        y_rng_r = y_max - coord_mean[1]
        z_rng_r = z_max - coord_mean[0]

        x_rng_l = coord_mean[2] - x_min
        y_rng_l = coord_mean[1] - y_min
        z_rng_l = coord_mean[0] - z_min

        #max_rng = max(max(x_rng, y_rng), z_rng)
        max_rng_r = max(max(x_rng_r, y_rng_r), z_rng_r)
        max_rng_l = max(max(x_rng_l, y_rng_l), z_rng_l)
        max_rng = max(max_rng_r, max_rng_l)

        Radius = max_rng

        max_rng = max_rng * (1 + margin_ratio)

        x_max = max(x_max,int(round(coord_mean[2]+max_rng)))
        x_min = min(x_min,int(round(coord_mean[2]-max_rng)))
        y_max = max(y_max,int(round(coord_mean[1]+max_rng)))
        y_min = min(y_min,int(round(coord_mean[1]-max_rng)))
        z_max = max(z_max,int(round(coord_mean[0]+max_rng)))
        z_min = min(z_min,int(round(coord_mean[0]-max_rng)))

        if z_min < 0:
            z_min = 0
        if y_min < 0:
            x_min = 0
        if x_min < 0:
            x_min = 0

        if z_max > np.shape(Input_vol)[0]:
            z_max = np.shape(Input_vol)[0]
        if y_max > np.shape(Input_vol)[1]:
            y_max = np.shape(Input_vol)[1]
        if x_max > np.shape(Input_vol)[2]:
            x_max = np.shape(Input_vol)[2]

        return Radius, x_max, x_min, y_max, y_min, z_max, z_min, x_rng, y_rng, z_rng

def resize_to_input(pixels, input):
    #print "input.shape, pixels.shape = ", input.shape, pixels.shape
    dsfactor = [w / float(f) for w, f in zip(input.shape, pixels.shape)]
    return scipy.ndimage.interpolation.zoom(pixels, dsfactor, mode="constant")

def getKey(item):
    return item[0]

def ND_Analysis_2D(nodule_Img):
    ND_Score = 0
    mask = Global_Thresholding_2D(nodule_Img)
    if np.amax(mask) >0:
        PCA_Score = regionprops(mask)[0].minor_axis_length/(regionprops(mask)[0].major_axis_length+0.0000000001)
        Circ_Score = regionprops(mask)[0].area/(np.pi*(regionprops(mask)[0].major_axis_length/2)*(regionprops(mask)[0].major_axis_length/2)+0.000000001)
        ND_Score = 0.7 * PCA_Score + 0.3 * Circ_Score
    else:
        ND_Score = 0
    return ND_Score

def make_diagonal_image(vol, FIX_LEN = 70):

    center = int(FIX_LEN/2)
    rotate = 45
    M = cv2.getRotationMatrix2D((center, center), rotate, 1)
    diagonal_vol = []
    for i in range(0, FIX_LEN):
        diagonal_vol.append(cv2.warpAffine(vol[i], M, (FIX_LEN, FIX_LEN), borderValue=-2000))

    diagonal_vol = np.array(diagonal_vol, dtype=np.float32)
    return np.transpose(diagonal_vol, (1, 2, 0))[center], np.transpose(diagonal_vol, (2, 1, 0))[center]


def Nodule_Candidate_Extract(Input_vol,lung_Seg_Vol, Mask_vol, patient_id,vol_Idx, nodule_total_array):


    Num_Label = 0
    #label_scan = label(Mask_vol)
    label_scan = Mask_vol
    areas = [r.area for r in regionprops(label_scan)]
    areas.sort()

    #print np.shape(areas)
    #nodule_total_array = []
    for region in regionprops(label_scan):
        Num_Label = Num_Label + 1
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

        data = np.array(points)
        coord_mean = []
        coord_mean.append(np.mean(coord_Z))
        coord_mean.append(np.mean(coord_Y))
        coord_mean.append(np.mean(coord_X))

        # sklearn_pca = sklearnPCA()
        # sklearn_pca.fit(data)

        R, x_max, x_min, y_max, y_min, z_max, z_min,  x_rng, y_rng, z_rng  = check_range(Input_vol, coord_X, coord_Y, coord_Z,
                                                                  coord_mean, 0.2)
        X_rng = (x_max - x_min) / 1.4
        Y_rng = (y_max - y_min) / 1.4
        Z_rng = (z_max - z_min) / 1.4
        # PCA_Score = region.area/(X_rng*Y_rng*Z_rng)

        max_rng = max(X_rng, Y_rng, Z_rng)
        min_rng = min(X_rng, Y_rng, Z_rng)

        #print max_rng
        if max_rng > 5 and max_rng < 60:

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

            #PCA_Score = (3 / 2) * (lamda2 + lamda3)
            # PCA_Score = abs(sklearn_pca.explained_variance_[2]) / (abs(sklearn_pca.explained_variance_[0]) + 0.00000000001)
            # Comp_Score = region.area/(4*np.pi*R*R*R/3)
            # ND_Score = 0.5*PCA_Score + 0.5*Comp_Score
            #
            # print PCA_Score
            # print sklearn_pca.explained_variance_[0], sklearn_pca.explained_variance_[1], sklearn_pca.explained_variance_[2]
            # if pca.s[0]>1000:
            nodule_vol = lung_Seg_Vol[int(z_min):int(z_max), int(y_min):int(y_max), int(x_min):int(x_max)]
            input_shape = np.zeros(shape=(70, 70, 70))
            vol_rescale = resize_to_input(nodule_vol, input_shape)
            nodule_vol = vol_rescale
            nodule_Img1 = nodule_vol[int(round(np.shape(nodule_vol)[0] / 2)),:,:]
            nodule_Img2 = nodule_vol[:, int(round(np.shape(nodule_vol)[0] / 2)),:]
            nodule_Img3 = nodule_vol[:, :, int(round(np.shape(nodule_vol)[0] / 2))]

            nodule_Img4, nodule_Img5 = make_diagonal_image(nodule_vol, np.shape(nodule_vol)[0])
            nodule_Img6, nodule_Img7 = make_diagonal_image(np.transpose(nodule_vol, (1, 2, 0)), np.shape(nodule_vol)[0])
            nodule_Img8, nodule_Img9 = make_diagonal_image(np.transpose(nodule_vol, (2, 1, 0)), np.shape(nodule_vol)[0])

            ND_Score1 = ND_Analysis_2D(nodule_Img1)
            ND_Score2 = ND_Analysis_2D(nodule_Img2)
            ND_Score3 = ND_Analysis_2D(nodule_Img3)
            ND_Score4 = ND_Analysis_2D(nodule_Img4)
            ND_Score5 = ND_Analysis_2D(nodule_Img5)
            ND_Score6 = ND_Analysis_2D(nodule_Img6)
            ND_Score7 = ND_Analysis_2D(nodule_Img7)
            ND_Score8 = ND_Analysis_2D(nodule_Img8)
            ND_Score9 = ND_Analysis_2D(nodule_Img9)


            temp_sc = []
            temp_sc.append(ND_Score1)
            temp_sc.append(ND_Score2)
            temp_sc.append(ND_Score3)
            temp_sc.append(ND_Score4)
            temp_sc.append(ND_Score5)
            temp_sc.append(ND_Score6)
            temp_sc.append(ND_Score7)
            temp_sc.append(ND_Score8)
            temp_sc.append(ND_Score9)

            #ND_Score = (np.amin(temp_sc)/(np.amax(temp_sc)+0.0000001))
            ND_Score = np.mean(temp_sc)



            nodule_total_array.append([ND_Score,x_max, x_min, y_max, y_min, z_max, z_min, x_rng,y_rng,z_rng, coord_mean[0], coord_mean[1], coord_mean[2]] )

    return nodule_total_array, Num_Label

def Nodule_Candidate_Save(Input_vol, patient_id, vol_Idx, nodule_total_array, writer):

    nodule_total_array.sort(key=lambda x: float(x[0]), reverse=True)
    # print nodule_total_array

    newpath1 = OUTPUT_FOLDER1 + "V" + str(vol_Idx) + "/"
    newpath2 = OUTPUT_FOLDER2 + "V" + str(vol_Idx) + "/"
    newpath3 = OUTPUT_FOLDER3 + "V" + str(vol_Idx) + "/"
    newpath4 = OUTPUT_FOLDER4 + "V" + str(vol_Idx) + "/"
    if not os.path.exists(newpath1):
        os.makedirs(newpath1)
    if not os.path.exists(newpath2):
        os.makedirs(newpath2)
    if not os.path.exists(newpath3):
        os.makedirs(newpath3)
    if not os.path.exists(newpath4):
        os.makedirs(newpath4)


    for Num_Label in xrange(np.shape(nodule_total_array)[0]):

        ND_Score = nodule_total_array[Num_Label][0]
        x_max = nodule_total_array[Num_Label][1]
        x_min = nodule_total_array[Num_Label][2]
        y_max = nodule_total_array[Num_Label][3]
        y_min = nodule_total_array[Num_Label][4]
        z_max = nodule_total_array[Num_Label][5]
        z_min = nodule_total_array[Num_Label][6]
        z_pos = nodule_total_array[Num_Label][10]
        y_pos = nodule_total_array[Num_Label][11]
        x_pos = nodule_total_array[Num_Label][12]
        nodule_vol = Input_vol[int(z_min):int(z_max), int(y_min):int(y_max), int(x_min):int(x_max)]
        # nodule_label = label_scan[int(z_min):int(z_max), int(y_min):int(y_max), int(x_min):int(x_max)]

        # print "lamda = ", sklearn_pca.explained_variance_[0],sklearn_pca.explained_variance_[1],sklearn_pca.explained_variance_[2], "max_range = ", np.shape(nodule_vol)[0], np.shape(nodule_vol)[1], np.shape(nodule_vol)[2]
        if (np.amin(np.shape(nodule_vol)) != 0):
            # print x_max, x_min, y_max, y_min, z_max, z_min
            # resize to 70,70,70
            input_shape = np.zeros(shape=(70, 70, 70))

            # print np.shape(nodule_vol)
            vol_rescale = resize_to_input(nodule_vol, input_shape)
            #nodule_label_rescale = resize_to_input(nodule_label, input_shape)

            # print np.amax(coord_Z), np.amin(coord_Z), int((np.amax(coord_Z) - np.amin(coord_Z)) / 2)
            # print np.amax(coord_Y), np.amin(coord_Y), int((np.amax(coord_Y) - np.amin(coord_Y)) / 2)
            # print np.amax(coord_X), np.amin(coord_X), int((np.amax(coord_X) - np.amin(coord_X)) / 2)
            # if np.amax(coord_Z) - np.amin(coord_Z) > 3:
            #     plt.figure(5)
            #     plt.imshow(nodule_vol[int((np.amax(coord_Z) - np.amin(coord_Z)) / 2)], cmap=plt.cm.gray)
            #     plt.figure(6)
            #     plt.imshow(Input_vol[int((np.amax(coord_Z) + np.amin(coord_Z)) / 2)], cmap=plt.cm.gray)
            #     plt.show()

            ##Save Nodule Candidate
            # for i in xrange(np.shape(vol_rescale)[0]):
            # save_volume(vol_rescale[i],"nodule_candidate/" + patient_id + "_ND" + str(Num_Label)+ "_Z" + str(i) + ".png" )
            writer.writerow({'V_Num':vol_Idx ,'P_ID':patient_id, 'N_ID':Num_Label, 'X_len': nodule_total_array[Num_Label][7],'Y_len': nodule_total_array[Num_Label][8], 'Z_len': nodule_total_array[Num_Label][9],
                             'X_pos': x_pos, 'Y_pos': y_pos, 'Z_pos': z_pos})

            # save_volume(vol_rescale[int(round(np.shape(vol_rescale)[0] / 2))],
            #             newpath1 + "V" + str(vol_Idx) + "_" + patient_id + "_ND" + str(Num_Label) + ".png")
            # save_volume(vol_rescale[:,int(round(np.shape(vol_rescale)[0] / 2)),:],
            #             newpath2 + "V" + str(vol_Idx) + "_" + patient_id + "_ND" + str(Num_Label) + ".png")
            # save_volume(vol_rescale[:,:,int(round(np.shape(vol_rescale)[0] / 2))],
            #             newpath3 + "V" + str(vol_Idx) + "_" + patient_id + "_ND" + str(Num_Label) + ".png")
            np.save(newpath4 + "V" + str(vol_Idx) + "_" + patient_id + "_ND" + str(Num_Label) + ".npy", vol_rescale)

            temp_vol_rescale = np.copy(vol_rescale)
            temp_vol_rescale[vol_rescale < -400] = 0
            temp_vol_rescale[vol_rescale >= -400] = 1

            # temp = Dot_Filtering(vol_rescale)
            # print "mean_dot=", np.mean(temp)

            # plt.figure(100)
            # plt.imshow(nodule_label_rescale[int(round(np.shape(nodule_label_rescale)[0]/2))], cmap=plt.cm.gray)
            # plt.figure(101)
            # #plt.imshow(np.max(vol_rescale, axis=0), cmap=plt.cm.gray)
            # plt.imshow(nodule_label_rescale[:,int(round(np.shape(nodule_label_rescale)[0]/2)),:], cmap=plt.cm.gray)
            # plt.figure(102)
            # #plt.imshow(np.max(vol_rescale[], axis=0), cmap=plt.cm.gray)
            # plt.imshow(nodule_label_rescale[:,:, int(round(np.shape(nodule_label_rescale)[0] / 2))], cmap=plt.cm.gray)
            # plt.figure(103)
            # plt.imshow(np.max(vol_rescale, axis=0), cmap=plt.cm.gray)
            # plt.show()
            # plot_3d(nodule_label, 0)

def Lung_3D_Filtering(segmented_ct_scan):

    Mask = np.zeros_like(segmented_ct_scan)
    Mask[segmented_ct_scan > -2000] = 1
    Mask[segmented_ct_scan == -2000] = 0
    Mask.astype(np.bool)
    result = np.zeros_like(segmented_ct_scan)-2000

    # print segmented_ct_scan
    label_image = label(Mask)
    areas = [r.area for r in regionprops(label_image)]
    areas.sort(reverse=True)
    # print np.shape(areas)[0], areas[0],areas[1],areas[2]
    if np.shape(areas)[0] > 1:
        for region in regionprops(label_image):
            if region.area == areas[0]:
                for coordinates in region.coords:
                    # print coordinates[0], coordinates[1], coordinates[2]
                    result[coordinates[0], coordinates[1], coordinates[2]] = segmented_ct_scan[
                        coordinates[0], coordinates[1], coordinates[2]]
        if float(areas[0]) / float(areas[1]) < 100:
            for region in regionprops(label_image):
                if region.area == areas[1]:
                    for coordinates in region.coords:
                        result[coordinates[0], coordinates[1], coordinates[2]] = segmented_ct_scan[
                            coordinates[0], coordinates[1], coordinates[2]]
    # plt.figure(100)
    # plt.imshow(result[200], cmap=plt.cm.gray)
    # plt.show()

    return result

#################################
##Main
#################################

patients = []
CSV_R_PATH = '/home/kshitij/Desktop/Dataset/stage2_sample_submission.csv'
CSV_W_PATH = '/home/kshitij/Desktop/Dataset/KAGGLE_Nodule_Param_VAL32.csv'
with open(CSV_R_PATH) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        patients.append(row['id'])
#patients = os.listdir(INPUT_FOLDER)
patients.sort()

with open(CSV_W_PATH,  'w') as csvfile:
    fieldnames = ['V_Num', 'P_ID', 'N_ID', 'X_len', 'Y_len', 'Z_len', 'X_pos', 'Y_pos', 'Z_pos']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    #print patients
    for i in xrange(len(patients)):
        #if i == 0:

        print "V"+str(i),INPUT_FOLDER+patients[i]
        images = load_scan(INPUT_FOLDER + patients[i])
        pixels = []
        for image in images:
            pixels.append(image.pixel_array)
        pixels = np.array(pixels, dtype=np.int16)

        minimum_pixel = np.amin(pixels[int(len(pixels) / 2)][0][0])


        pixels_array = []
        for k in range(0,len(images)):
            pixel = get_pixels_hu(images, pixels[k], minimum_pixel, k)
            pixels_array.append(pixel)


        pixels_array = np.array(pixels_array, dtype=np.int16)

        start_time = time.time()
        patient_pixels = pixels_array

        pix_resampled, spacing = resample(patient_pixels, images, [1, 1, 1])
        print("--- resample %s seconds ---" % (time.time() - start_time))

        print("Shape before resampling\t", patient_pixels.shape)
        print("Shape after resampling\t", pix_resampled.shape)

        start_time = time.time()
        #
        segmented_ct_scan = segment_lung_from_ct_scan(pix_resampled)


        segmented_ct_scan = Lung_3D_Filtering(segmented_ct_scan)
        #segmented_ct_scan = get_segmented_lungs_Enhance(pix_resampled)
        print("--- segment_lung_from_ct_scan %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
        segmented_ct_scan_Blur = ndimage.gaussian_filter(segmented_ct_scan, sigma=0.5)
        print("--- gaussian_filter %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
        Nodule_Candidate = Global_Thresholding_3D(segmented_ct_scan_Blur)
        print("--- Global_Thresholding_3D %s seconds ---" % (time.time() - start_time))

        start_time = time.time()
        selem = disk(3)
        Nodule_Candidate_Op1 = binary_opening_3D(Nodule_Candidate, selem)
        # selem = disk(2)
        # Nodule_Candidate_Op2 = binary_opening_3D(Nodule_Candidate, selem)
        # selem = disk(3)
        # Nodule_Candidate_Op3 = binary_opening_3D(Nodule_Candidate, selem)
        print("--- binary_opening_3D %s seconds ---" % (time.time() - start_time))
        #
        # plt.figure(4)
        # plt.imshow(Nodule_Candidate[SN], cmap=plt.cm.gray)
        # plt.show()

        # for i in xrange(235,250):
        #     plt.figure(5)
        #     plt.imshow(segmented_ct_scan_Blur[i], cmap=plt.cm.gray)
        #     plt.figure(6)
        #     plt.imshow(Nodule_Candidate[i], cmap=plt.cm.gray)
        #     # plt.figure(7)
        #     # plt.imshow(Nodule_Candidate_mor[i], cmap=plt.cm.gray)
        #     plt.show()

        start_time = time.time()
        Total_Num_Label = 0
        nodule_total_array = []

        distance1 = ndi.distance_transform_edt(Nodule_Candidate)
        local_maxi1 = Dist_Thresholding_3D(distance1)
        markers1 = label(local_maxi1)
        labels1 = watershed(-distance1, markers1, mask=Nodule_Candidate)
        nodule_total_array, Total_Num_Label1 = Nodule_Candidate_Extract(pix_resampled,segmented_ct_scan, labels1, patients[i], i, nodule_total_array)

        Total_Num_Label = Total_Num_Label + Total_Num_Label1

        distance2 = ndi.distance_transform_edt(Nodule_Candidate_Op1)
        local_maxi2 = Dist_Thresholding_3D(distance2)
        markers = label(local_maxi2)
        labels2 = watershed(-distance2, markers, mask=Nodule_Candidate_Op1)
        nodule_total_array, Total_Num_Label2 = Nodule_Candidate_Extract(pix_resampled, segmented_ct_scan, labels2, patients[i], i,nodule_total_array)


        Total_Num_Label = Total_Num_Label + Total_Num_Label2

        print nodule_total_array

        Nodule_Candidate_Save(pix_resampled,   patients[i], i, nodule_total_array, writer)
        print "Total_Num_Label=", Total_Num_Label
        print("--- Nodule_Candidate_Extract %s seconds ---" % (time.time() - start_time))
        # for k in xrange(200,700,2):
        #     plt.figure(5)
        #     plt.imshow(segmented_ct_scan_Blur[k],cmap=plt.cm.gray)
        #     # plt.figure(6)
        #     # plt.imshow(NoduleMask1[k], cmap='gray', vmin=0, vmax=255)
        #     # plt.figure(7)
        #     # plt.imshow(NoduleMask2[k], cmap='gray', vmin=0, vmax=255)
        #     # plt.figure(8)
        #     # plt.imshow(NoduleMask3[k], cmap='gray', vmin=0, vmax=255)
        #     # plt.figure(9)
        #     # plt.imshow(NoduleMask_bool[k], cmap=plt.cm.gray)
        #     plt.figure(10)
        #     plt.imshow(labels1[k],  cmap='spectral', interpolation='nearest')
        #     plt.figure(11)
        #     plt.imshow(labels2[k], cmap='spectral', interpolation='nearest')
        #     plt.show()






                #plot_3d(Nodule_Candidate, 0)
    print "preprocessing completed."






