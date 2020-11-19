### Module in order to preprocess the satellite images and ready them for input into deep learning models
### Max Zvyagin

import rasterio
import rasterio.features
import geopandas as gpd
import torch
from torch.utils.data import Dataset
import math
import numpy as np
from skimage.color import rgb2hsv
import pickle
from os import path
import sys
import random
# import imgaug as ia
# import imgaug.augmenters as iaa
# from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import tensorflow as tf
from sklearn.model_selection import train_test_split


def mask_from_shp(img_f, shp_f):
    # this function will also perform the reprojection onto the image file that we're masking (CRS adjustment)
    # read in the shp file
    shp = gpd.read_file(shp_f)
    img = rasterio.open(img_f)
    # get the crs from the image file
    new_crs = str(img.crs).lower()
    # perform the reprojection
    shp_reproject = shp.to_crs(new_crs)
    # now that the shapes are lined up, get the mask from the .shp geometry
    geometry = shp_reproject['geometry']
    mask = rasterio.features.geometry_mask(geometry, img.shape, img.transform, all_touched=False, invert=True)
    mask = mask.astype(float)
    # mask[mask == 1] = 255
    return mask


def mask_from_output(model_output):
    # given model output, softmax probability for 2 classes, generate a mask corresponding to segmentation
    # get final shape of output
    final_shape = model_output.shape[-2:]
    print(final_shape)
    result = []
    channel_one = torch.flatten(model_output[0])
    channel_two = torch.flatten(model_output[1])
    for i in range(len(torch.flatten(model_output[0]))):
        if channel_one[i] >= channel_two[i]:
            result.append(0)
        else:
            result.append(1)
    result = torch.reshape(torch.FloatTensor(result), final_shape).unsqueeze(0)
    return result


def split(array):
    # split a given 3d array into 4 equal chunks
    if len(array.shape) == 2:
        mid_x = int(array.shape[0] / 2)
        mid_y = int(array.shape[1] / 2)
        first = array[:mid_x, :mid_y]
        second = array[mid_x:, :mid_y]
        third = array[:mid_x, mid_y:]
        fourth = array[mid_x:, mid_y:]
    else:
        mid_x = int(array.shape[1] / 2)
        mid_y = int(array.shape[2] / 2)
        first = array[:, :mid_x, :mid_y]
        second = array[:, mid_x:, :mid_y]
        third = array[:, :mid_x, mid_y:]
        fourth = array[:, mid_x:, mid_y:]
    chunks = [first, second, third, fourth]
    return chunks


def check_if_good_sample(mask_sample):
    # num_pos = np.count_nonzero(mask_sample == 255)
    num_pos = np.count_nonzero(mask_sample)
    # only collect as a sample if it makes up at least 10 percent of the image
    if num_pos / mask_sample.size >= .05:
        return True
    else:
        return False


# given the name of an image file and the corresponding .shp array mask, outputs an array of image windows and mask windows
def get_windows(img_f, mask, large_image=False, unlabelled=False, num=500, get_max=True, rand=False):
    samples = []
    with rasterio.open(img_f) as src:
        image = src.block_windows()
        if rand:
            random.shuffle(list(image))
        for ji, window in image:
            if len(samples) >= num and not get_max:
                return samples
            # get the window from the mask
            mask_check = mask[window.row_off:window.row_off + window.height,
                         window.col_off:window.col_off + window.width]
            if check_if_good_sample(mask_check):
                if not unlabelled:
                    r = src.read(window=window)
                    if large_image:
                        samples.append((torch.from_numpy(r).float(), torch.from_numpy(mask_check).float()))
                    else:
                        # need to split into tiles
                        r_chunks = split(r)
                        mask_chunks = split(mask_check)
                        for i in range(4):
                            samples.append(
                                (torch.from_numpy(r_chunks[i]).float(), torch.from_numpy(mask_chunks[i]).float()))
            else:
                if unlabelled:
                    r = src.read(window=window)
                    if large_image:
                        chunk_size = r.shape
                        if chunk_size[-1] == 512 and chunk_size[-2] == 512:
                            samples.append(torch.from_numpy(r).float())
                    else:
                        # need to split into tiles
                        r_chunks = split(r)
                        for i in range(4):
                            chunk_size = r_chunks[i].shape
                            if chunk_size[-1] == 256 and chunk_size[-2] == 256:
                                samples.append(torch.from_numpy(r_chunks[i]).float())
    return samples


# return 3 channel image of rgb reflectance values
def get_rgb_windows(img_f, mask, large_image=False, unlabelled=False, num=500, get_max=True, rand=False):
    samples = []
    with rasterio.open(img_f) as src:
        if rand:
            image = random.shuffle(list(src.block_windows()))
        else:
            image = src.block_windows()
        for ji, window in image:
            if len(samples) >= num and not get_max:
                return samples
            # get the window from the mask
            mask_check = mask[window.row_off:window.row_off + window.height,
                         window.col_off:window.col_off + window.width]
            if check_if_good_sample(mask_check):
                if not unlabelled:
                    r = src.read(window=window)
                    if large_image:
                        samples.append((torch.from_numpy(r[:3]).float(), torch.from_numpy(mask_check).float()))
                    else:
                        # need to split into tiles
                        r_chunks = split(r)
                        mask_chunks = split(mask_check)
                        for i in range(4):
                            samples.append(
                                (torch.from_numpy(r_chunks[i][:3]).float(), torch.from_numpy(mask_chunks[i]).float()))
            else:
                if unlabelled:
                    r = src.read(window=window)
                    if large_image:
                        chunk_size = r.shape
                        if chunk_size[-1] == 512 and chunk_size[-2] == 512:
                            samples.append(torch.from_numpy(r[:3]).float())
                    else:
                        # need to split into tiles
                        r_chunks = split(r)
                        for i in range(4):
                            chunk_size = r_chunks[i].shape
                            if chunk_size[-1] == 256 and chunk_size[-2] == 256:
                                samples.append(torch.from_numpy(r_chunks[i][:3]).float())
    return samples


# return single channel image of solely infrared reflectance values
# @lru_cache(maxsize=2)
def get_ir_windows(img_f, mask, large_image=False, unlabelled=False, num=500, get_max=True, rand=False):
    samples = []
    with rasterio.open(img_f) as src:
        if rand:
            image = random.shuffle(list(src.block_windows()))
        else:
            image = src.block_windows()
        for ji, window in image:
            if len(samples) >= num and not get_max:
                return samples
            # get the window from the mask
            mask_check = mask[window.row_off:window.row_off + window.height,
                         window.col_off:window.col_off + window.width]
            if check_if_good_sample(mask_check):
                if not unlabelled:
                    r = src.read(window=window)
                    if large_image:
                        samples.append((torch.from_numpy(r[3]).float(), torch.from_numpy(mask_check).float()))
                    else:
                        # need to split into tiles
                        r_chunks = split(r)
                        mask_chunks = split(mask_check)
                        for i in range(4):
                            samples.append(
                                (torch.from_numpy(r_chunks[i][3]).float(), torch.from_numpy(mask_chunks[i]).float()))
            else:
                if unlabelled:
                    r = src.read(window=window)
                    if large_image:
                        chunk_size = r.shape
                        if chunk_size[-1] == 512 and chunk_size[-2] == 512:
                            samples.append(torch.from_numpy(r[3]).float())
                    else:
                        # need to split into tiles
                        r_chunks = split(r)
                        for i in range(4):
                            chunk_size = r_chunks[i].shape
                            if chunk_size[-1] == 256 and chunk_size[-2] == 256:
                                samples.append(torch.from_numpy(r_chunks[i][3]).float())
    return samples


# return 3 channels of rgb converted to hsv
# @lru_cache(maxsize=2)
def get_hsv_windows(img_f, mask, large_image=False, unlabelled=False, num=500, get_max=True, rand=False):
    samples = []
    with rasterio.open(img_f) as src:
        if rand:
            image = random.shuffle(list(src.block_windows()))
        else:
            image = src.block_windows()
        for ji, window in image:
            if len(samples) >= num and not get_max:
                return samples
            # get the window from the mask
            mask_check = mask[window.row_off:window.row_off + window.height,
                         window.col_off:window.col_off + window.width]
            if check_if_good_sample(mask_check):
                if not unlabelled:
                    r = src.read(window=window)
                    if large_image:
                        new_val = rgb2hsv(np.moveaxis(r[:3], 0, -1))
                        new_val = np.moveaxis(new_val, -1, 0)
                        samples.append((torch.from_numpy(new_val).float(), torch.from_numpy(mask_check).float()))
                    else:
                        # need to split into tiles
                        r_chunks = split(r)
                        mask_chunks = split(mask_check)
                        for i in range(4):
                            new_val = rgb2hsv(np.moveaxis(r_chunks[i][:3], 0, -1))
                            new_val = np.moveaxis(new_val, -1, 0)
                            samples.append(
                                (torch.from_numpy(new_val).float(), torch.from_numpy(mask_chunks[i]).float()))
            else:
                if unlabelled:
                    r = src.read(window=window)
                    if large_image:
                        chunk_size = r.shape
                        if chunk_size[-1] == 512 and chunk_size[-2] == 512:
                            new_val = rgb2hsv(np.moveaxis(r[:3], 0, -1))
                            new_val = np.moveaxis(new_val, -1, 0)
                            samples.append(torch.from_numpy(new_val).float())
                    else:
                        # need to split into tiles
                        r_chunks = split(r)
                        for i in range(4):
                            chunk_size = r_chunks[i].shape
                            if chunk_size[-1] == 256 and chunk_size[-2] == 256:
                                new_val = rgb2hsv(np.moveaxis(r_chunks[i][:3], 0, -1))
                                new_val = np.moveaxis(new_val, -1, 0)
                                samples.append(torch.from_numpy(new_val).float())
    return samples


# return rgb converted to hsv in addition to infrared channel
def get_hsv_with_ir_windows(img_f, mask, large_image=False, unlabelled=False, num=500, get_max=True, rand=False):
    samples = []
    with rasterio.open(img_f) as src:
        if rand:
            image = random.shuffle(list(src.block_windows()))
        else:
            image = src.block_windows()
        for ji, window in image:
            if len(samples) >= num and not get_max:
                return samples
            # get the window from the mask
            mask_check = mask[window.row_off:window.row_off + window.height,
                         window.col_off:window.col_off + window.width]
            if check_if_good_sample(mask_check):
                if not unlabelled:
                    r = src.read(window=window)
                    if large_image:
                        new_val = rgb2hsv(np.moveaxis(r[:3], 0, -1))
                        new_val = np.moveaxis(new_val, -1, 0)
                        all_channels = np.concatenate((new_val, np.expand_dims(r[3], 0)), axis=0)
                        samples.append((torch.from_numpy(all_channels).float(), torch.from_numpy(mask_check).float()))
                    else:
                        # need to split into tiles
                        r_chunks = split(r)
                        mask_chunks = split(mask_check)
                        for i in range(4):
                            new_val = rgb2hsv(np.moveaxis(r_chunks[i][:3], 0, -1))
                            new_val = np.moveaxis(new_val, -1, 0)
                            all_channels = np.concatenate((new_val, np.expand_dims(r_chunks[i][3], 0)), axis=0)
                            samples.append(
                                (torch.from_numpy(all_channels).float(), torch.from_numpy(mask_chunks[i]).float()))
        else:
            if unlabelled:
                r = src.read(window=window)
                if large_image:
                    chunk_size = r.shape
                    if chunk_size[-1] == 512 and chunk_size[-2] == 512:
                        new_val = rgb2hsv(np.moveaxis(r[:3], 0, -1))
                        new_val = np.moveaxis(new_val, -1, 0)
                        all_channels = np.concatenate((new_val, np.expand_dims(r[3], 0)), axis=0)
                        samples.append(torch.from_numpy(all_channels).float())
                else:
                    # need to split into tiles
                    r_chunks = split(r)
                    for i in range(4):
                        chunk_size = r_chunks[i].shape
                        if chunk_size[-1] == 256 and chunk_size[-2] == 256:
                            new_val = rgb2hsv(np.moveaxis(r_chunks[i][:3], 0, -1))
                            new_val = np.moveaxis(new_val, -1, 0)
                            all_channels = np.concatenate((new_val, np.expand_dims(r_chunks[i][3], 0)), axis=0)
                            samples.append(torch.from_numpy(all_channels).float())
    return samples


# given the name of an image file and the corresponding .shp array mask, outputs an array of calculated vegetation index values and mask
def get_vegetation_index_windows(img_f, mask, large_image=False, unlabelled=False, num=500, get_max=True, rand=False):
    samples = []
    with rasterio.open(img_f) as src:
        if rand:
            image = random.shuffle(list(src.block_windows()))
        else:
            image = src.block_windows()
        for ji, window in image:
            if len(samples) >= num and not get_max:
                return samples
            # get the window from the mask
            mask_check = mask[window.row_off:window.row_off + window.height,
                         window.col_off:window.col_off + window.width]
            if check_if_good_sample(mask_check):
                if not unlabelled:
                    r = src.read(2, window=window)
                    i = src.read(3, window=window)
                    veg = numpy_msavi(r, i)
                    if large_image:
                        samples.append((torch.from_numpy(veg).float(), torch.from_numpy(mask_check).float()))
                    else:
                        chunks = split(veg)
                        mask_chunks = split(mask_check)
                        # the split function return 4 separate quadrants from the original window
                        for i in range(4):
                            samples.append((torch.from_numpy(chunks[i]).float(),
                                            torch.from_numpy(mask_chunks[i]).float()))
            else:
                if unlabelled:
                    r = src.read(2, window=window)
                    i = src.read(3, window=window)
                    veg = numpy_msavi(r, i)
                    if large_image:
                        chunk_size = veg.shape
                        if chunk_size[-1] == 512 and chunk_size[-2] == 512:
                            samples.append(torch.from_numpy(veg).float())
                    else:
                        r_chunks = split(veg)
                        # the split function return 4 separate quadrants from the original window
                        for i in range(4):
                            chunk_size = r_chunks[i].shape
                            if chunk_size[-1] == 256 and chunk_size[-2] == 256:
                                samples.append(torch.from_numpy(chunks[i]).float())
                pass
    return samples


# given red and infrared reflectance values, calculate the vegetation index (desert version from Yuki's paper)
def msavi(red, infrared):
    return (2 * infrared) + 1 - math.sqrt((2 * infrared + 1) ** 2 - (8 * (infrared - red))) / 2


numpy_msavi = np.vectorize(msavi)


# def augment_dataset(dataset):
#     # generate augmented samples of dataset
#     ia.seed(1)
#     # flip from left to right
#     seq = iaa.Sequential([iaa.Fliplr()])
#     augmented_samples = []
#     for sample in dataset:
#         img = sample['image'].numpy()
#         img = np.moveaxis(img, 0, -1)
#         seg = SegmentationMapsOnImage(sample['mask'].numpy().astype(bool), shape=img.shape)
#         i, s = seq(image=img, segmentation_maps=seg)
#         s = torch.FloatTensor(s.get_arr().copy())
#         i = torch.FloatTensor(np.moveaxis(i, -1, 0).copy())
#         augmented_samples.append((i, s))
#     # do the same thing but flip images upside down
#     seq = iaa.Sequential([iaa.Flipud()])
#     for sample in dataset:
#         img = sample['image'].numpy()
#         img = np.moveaxis(img, 0, -1)
#         seg = SegmentationMapsOnImage(sample['mask'].numpy().astype(bool), shape=img.shape)
#         i, s = seq(image=img, segmentation_maps=seg)
#         s = torch.FloatTensor(s.get_arr().copy())
#         i = torch.FloatTensor(np.moveaxis(i, -1, 0).copy())
#         augmented_samples.append((i, s))
#     return augmented_samples


def pt_gis_train_test_split(img_and_shps=None, image_type="full_channel", large_image=False, theta=True):
    """ Return PT GIS Datasets with Train Test Split"""

    if not img_and_shps:
        img_and_shps = [("/scratch/mzvyagin/Ephemeral_Channels/Imagery/vhr_2012_refl.img",
                         "/scratch/mzvyagin/Ephemeral_Channels/Reference/reference_2012_merge.shp"),
                        ("/scratch/mzvyagin/Ephemeral_Channels/Imagery/vhr_2014_refl.img",
                         "/scratch/mzvyagin/Ephemeral_Channels/Reference/reference_2014_merge.shp")]

    samples = []
    for pair in img_and_shps:
        name = "/tmp/mzvyagin/"
        name += "gis_data"
        name += image_type
        if large_image:
            name += "large_image"
        name += "PTdataset.pkl"
        if path.exists(name):
            try:
                cache_object = open(name, "rb")
                train, test = pickle.load(cache_object)
                return PT_GISDataset(train), PT_GISDataset(test)
            except:
                print("ERROR: could not load from cache file. Please try removing " + name + " and try again.")
                sys.exit()
        # process each pair and generate the windows
        else:
            mask = mask_from_shp(pair[0], pair[1])
            if image_type == "full_channel":
                windows = get_windows(pair[0], mask, large_image)
            elif image_type == "rgb":
                windows = get_rgb_windows(pair[0], mask, large_image)
            elif image_type == "ir":
                windows = get_ir_windows(pair[0], mask, large_image)
            elif image_type == "hsv":
                windows = get_hsv_windows(pair[0], mask, large_image)
            elif image_type == "hsv_with_ir":
                windows = get_hsv_with_ir_windows(pair[0], mask, large_image)
            elif image_type == "veg_index":
                windows = get_vegetation_index_windows(pair[0], mask, large_image)
            else:
                print("WARNING: no image type match, defaulting to RGB+IR")
                windows = get_windows(pair[0], mask, large_image)
            # cache the windows
        samples.extend(windows)
        # now create test train split of samples
        train, test = train_test_split(samples, train_size=0.8, shuffle=False, random_state=0)
        cache_object = open(name, "wb")
        pickle.dump((train, test), cache_object)
        return PT_GISDataset(train), PT_GISDataset(test)


class PT_GISDataset(Dataset):
    """Generates a dataset for Pytorch of image and labelled mask."""

    # need to be given a list of tuple consisting of filepaths, (img, shp) to get pairs of windows for training
    def __init__(self, data_list):
        # can be initialized from a list of samples instead of from files
        self.samples = data_list

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def pt_to_tf(x):
    """ Converts a pytorch tensor to a tensorflow tensor and returns it"""
    n = x.numpy()
    n = np.swapaxes(n, 0, -1)
    t = tf.convert_to_tensor(n)
    return t


def tf_gis_test_train_split(img_and_shps=None, image_type="full_channel", large_image=False, theta=True):
    """ Returns a Tensorflow dataset of images and masks"""
    # Default is theta file system location
    if not img_and_shps:
        img_and_shps = [("/scratch/mzvyagin/Ephemeral_Channels/Imagery/vhr_2012_refl.img",
                         "/scratch/mzvyagin/Ephemeral_Channels/Reference/reference_2012_merge.shp"),
                        ("/scratch/mzvyagin/Ephemeral_Channels/Imagery/vhr_2014_refl.img",
                         "/scratch/mzvyagin/Ephemeral_Channels/Reference/reference_2014_merge.shp")]

    x_samples, y_samples = [], []
    for pair in img_and_shps:
        name = "/tmp/mzvyagin/"
        name += "gis_data"
        name += image_type
        if large_image:
            name += "large_image"
        name += "TFdataset.pkl"
        if path.exists(name):
            try:
                cache_object = open(name, "rb")
                (x_train, y_train), (x_test, y_test) = pickle.load(cache_object)
                return (x_train, y_train), (x_test, y_test)
            except:
                print("ERROR: could not load from cache file. Please try removing " + name + " and try again.")
                sys.exit()
        # process each pair and generate the windows
        else:
            mask = mask_from_shp(pair[0], pair[1])
            if image_type == "full_channel":
                windows = get_windows(pair[0], mask, large_image)
            elif image_type == "rgb":
                windows = get_rgb_windows(pair[0], mask, large_image)
            elif image_type == "ir":
                windows = get_ir_windows(pair[0], mask, large_image)
            elif image_type == "hsv":
                windows = get_hsv_windows(pair[0], mask, large_image)
            elif image_type == "hsv_with_ir":
                windows = get_hsv_with_ir_windows(pair[0], mask, large_image)
            elif image_type == "veg_index":
                windows = get_vegetation_index_windows(pair[0], mask, large_image)
            else:
                print("WARNING: no image type match, defaulting to RGB+IR")
                windows = get_windows(pair[0], mask, large_image)
            # cache the windows
            # need to convert to the tensorflow tensors instead of pytorch
            x, y = [], []
            for sample in windows:
                x.append(pt_to_tf(sample[0]))
                y.append(pt_to_tf(sample[1]))
            x_samples.extend(x)
            y_samples.extend(y)

    # generate test_train splits
    x_train, x_test, y_train, y_test = train_test_split(x_samples, y_samples, train_size=0.8, shuffle=False,
                                                        random_state=0)
    cache_object = open(name, "wb")
    #train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    #test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    pickle.dump(((x_train, y_train), (x_test, y_test)), cache_object)
    return (x_train, y_train), (x_test, y_test)
