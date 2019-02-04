import sys
import os
import matplotlib.image as mpimg
import numpy as np
from PIL import Image

# Dimensions of input images
IMG_WIDTH = 200
IMG_HEIGHT = 200

IMG_PATCH_SIZE = 8
IMG_CONTEXT_SIZE = 64
IMG_BORDER_SIZE = int((IMG_CONTEXT_SIZE - IMG_PATCH_SIZE) / 2)
IMG_PATCH_STRIDE = 4

OBJECTS_PATH = "../objects/"
PATCHES_MEAN_PATH = OBJECTS_PATH + "patches_mean"
RESULTS_PATH = "../results"
  
# roate by 45 degrees
def rotate_45(switch,image):

    if switch==1: # input is image and not label
        colorImage  = Image.fromarray(np.uint8(image*255))
        rotated     = colorImage.rotate(45)
        rot=np.asarray(rotated)

    else:
        temp_img_188=np.uint8(image)
        temp_img_881=temp_img_188.transpose()
        temp_img_881=temp_img_881*255

        # decreasing one dimension
        temp_img_88=np.zeros((8,8))
        for i in range(8):
            for j in range(8):
                temp_img_88[i][j]=temp_img_881[i][j][0]

        colorImage  = Image.fromarray(temp_img_88)
        rotated     = colorImage.rotate(45)
        rot_88=np.asarray(rotated)

        # increasing one dimension
        rot_188=np.zeros((1,8,8))
        for i in range(8):
            for j in range(8):
                rot_188[0][i][j]=rot_88[i][j]

        rot=rot_188

    return rot

def zero_center(patches):
    #Zero centers patch data and caches their mean value to disk

    if 1:
#    if os.path.isfile(PATCHES_MEAN_PATH + ".npy"):
#        mean_patch = np.load(PATCHES_MEAN_PATH + ".npy")  
#    else:
#        if not os.path.isdir(OBJECTS_PATH):            
#            os.makedirs(OBJECTS_PATH)
        print("Computing mean patch")

        full_length=len(patches)
        quarter_length=int(full_length/4)

        patches1=[]
        patches2=[]
        patches3=[]
        patches4=[]
        patches1=patches[0:quarter_length]
        patches2=patches[quarter_length:2*quarter_length]
        patches3=patches[2*quarter_length:3*quarter_length]
        patches4=patches[3*quarter_length:full_length]

        mean_patch1 = np.mean(patches1, axis=0)
        mean_patch2 = np.mean(patches2, axis=0)
        mean_patch3 = np.mean(patches3, axis=0)
        mean_patch4 = np.mean(patches4, axis=0)

        print("Mean computed")
        zero_centered_patches = patches - mean_patch1 - mean_patch2 - mean_patch3 - mean_patch4

        patches1 = None
        patches2=None
        patches3=None
        patches4=None

        mean_patch1 = None
        mean_patch2 = None
        mean_patch3 = None
        mean_patch4 = None

    return zero_centered_patches
    
def augment_image(switch,img, out_ls, num_of_transformations):
    #Augments the input image img by a number of transformations (rotations by 90° and flips). 

    # img1 is flipped image and img2 is rotated img. Both are augmented    

    img1 = np.fliplr(img)
    out_ls.append(img1)
    
    img2=rotate_45(switch,img)
    out_ls.append(img2)

    if num_of_transformations > 0:
        tmp = np.rot90(img)
        out_ls.append(tmp)
    if num_of_transformations > 1:
        tmp = np.rot90(np.rot90(img))
        out_ls.append(tmp)
    if num_of_transformations > 2:
        tmp = np.rot90(np.rot90(np.rot90(img)))
        out_ls.append(tmp)
    
    if num_of_transformations > 3:
        tmp = np.rot90(img2)
        out_ls.append(tmp)
    if num_of_transformations > 4:
        tmp = np.rot90(np.rot90(img2))
        out_ls.append(tmp)
    if num_of_transformations > 5:
        tmp = np.rot90(np.rot90(np.rot90(img2)))
        out_ls.append(tmp)
    if num_of_transformations > 6:
        out_ls.append(img2)
        
def mirror_border(img, border_size):
    #Pads an input image img with a border of size border_size using a mirror boundary condition
    
    if len(img.shape) < 3:
        # Binary image
        res_img = np.zeros((img.shape[0] + 2 * border_size, img.shape[1] + 2 * border_size))
    else:
        # 3 channel image
        res_img = np.zeros((img.shape[0] + 2 * border_size, img.shape[1] + 2 * border_size, 3))
    for i in range(border_size):
        res_img[border_size : res_img.shape[0] - border_size, border_size - 1 - i] = img[:, i]                                     # left columns
        res_img[border_size : res_img.shape[0] - border_size, res_img.shape[1] - border_size + i] = img[:, img.shape[1] - 1 - i]   # right columns
        res_img[border_size - 1 - i, border_size : res_img.shape[1] - border_size] = img[i, :]                                     # top rows
        res_img[res_img.shape[0] - border_size + i, border_size : res_img.shape[1] - border_size] = img[img.shape[0] - 1 - i, :]   # bottom rows
    res_img[border_size : res_img.shape[0] - border_size, border_size : res_img.shape[1] - border_size] = np.copy(img)
    # Corners
    res_img[0 : border_size, 0 : border_size] = \
        np.fliplr(np.flipud(img[0 : border_size, 0 : border_size]))
    res_img[0 : border_size, res_img.shape[1] - border_size : res_img.shape[1]] = \
        np.fliplr(np.flipud(img[0 : border_size, img.shape[1] - border_size : img.shape[1]]))
    res_img[res_img.shape[0] - border_size : res_img.shape[0], 0 : border_size] = \
        np.fliplr(np.flipud(img[img.shape[0] - border_size : img.shape[0], 0 : border_size]))
    res_img[res_img.shape[0] - border_size : res_img.shape[0], res_img.shape[1] - border_size : res_img.shape[1]] = \
        np.fliplr(np.flipud(img[img.shape[0] - border_size : img.shape[0], img.shape[1] - border_size : img.shape[1]])) 

    return res_img

def img_crop(switch,im, patch_size, border_size, stride, num_of_transformations):
    # Extracts patches of size patch_size and stride stride from an image img """
    
    context_size = patch_size + 2 * border_size    
    im = mirror_border(im, border_size)
    
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_single_channel = len(im.shape) < 3

    for i in range(0, imgheight - context_size + 1, stride):
        for j in range(0, imgwidth - context_size + 1, stride):
            if is_single_channel:
                # [1, patch_size, patch_size]
                im_patch = [im[j:j+context_size, i:i+context_size]]
            else:
                # [patch_size, patch_size, num_of_channels]
                im_patch = im[j:j+context_size, i:i+context_size, :]

            augment_image(switch,im_patch, list_patches, num_of_transformations)

    return list_patches    

def input_img_crop(switch,im, patch_size, border_size, stride, num_of_transformations):
    #Crops an input image. Direct alias of img_crop
    return img_crop(switch,im, patch_size, border_size, stride, num_of_transformations)
    

def label_img_crop(switch,im, patch_size, stride, num_of_transformations):
    #Crops a label image into patches    
    return img_crop(switch,im, patch_size, 0, stride, num_of_transformations)

def extract_data(filename_base, num_images, num_of_transformations=6, patch_size=IMG_PATCH_SIZE,
                 patch_stride=IMG_PATCH_STRIDE):
    #Extract patches from images
    imgs = []
    for i in range(1, num_images+1):
        imageid = "satImage_%.3d" % i
        image_filename = filename_base + imageid + ".png"
        if os.path.isfile(image_filename):
            print('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            imgs.append(img)
        else:
            print('File ' + image_filename + ' does not exist')

    num_images = len(imgs)
    print('Extracting patches...')

    data = []
    for i in range(num_images):
        this_img_patches = input_img_crop(1,imgs[i], patch_size, IMG_BORDER_SIZE, patch_stride,
                                              num_of_transformations)
        data += this_img_patches
        print("\tcurrently have %d patches" % len(data))
    
    print(str(len(data)) + ' patches extracted.')    
    imgs = None
    print("Casting to numpy array")
    tmp = np.asarray(data)
    data = None
    print("Cast successful")
    patches = zero_center(tmp)
    print("Patches have been zero centered.")

    return patches

def value_to_class(v):
    #Assign a label to a patch v
    foreground_threshold = 0.25  # percentage of pixels > 1 required to assign a foreground label to a patch
    df = np.sum(v)
    if df > foreground_threshold:
        # Road
        return [0, 1]
    else:
        # Non-road
        return [1, 0]

def extract_labels(filename_base, num_images, num_of_transformations=6, patch_size=IMG_PATCH_SIZE,
                   patch_stride=IMG_PATCH_STRIDE):
    #Extract the labels into a 1-hot matrix [image index, label index]
    gt_imgs = []
    for i in range(1, num_images+1):
        imageid = "satImage_%.3d" % i
        image_filename = filename_base + imageid + ".png"
        if os.path.isfile(image_filename):
            print('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            gt_imgs.append(img)
        else:
            print('File ' + image_filename + ' does not exist')

    num_images = len(gt_imgs)
    print('Extracting patches...')

    gt_patches = [label_img_crop(0,gt_imgs[i], patch_size, patch_stride, num_of_transformations)
                  for i in range(num_images)]
    data = np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])

    labels = np.asarray([value_to_class(np.mean(data[i])) for i in range(len(data))])
    print(str(len(data)) + ' label patches extracted.')

    # Convert to dense 1-hot representation.
    return labels.astype(np.float32)    

def pixel_to_patch_labels(im, patch_size, stride):
    #Convert 1-hot pixel-wise labels to a low-resolution image of patch labels
    imgwidth = im.shape[0]
    imgheight = im.shape[1]

    outW = 0
    outH = 0
    for i in range(0, imgheight - patch_size + 1, stride):
        outH += 1
    for j in range(0, imgwidth - patch_size + 1, stride):
        outW += 1
       
    output = np.zeros((outW, outH))
    idxI = 0
    for i in range(0, imgheight - patch_size + 1, stride):
        idxJ = 0
        for j in range(0, imgwidth - patch_size + 1, stride):
            im_patch = [im[j:j+patch_size, i:i+patch_size]]
            output[idxJ, idxI] = value_to_class(np.mean(im_patch))[1]
            idxJ += 1
        idxI += 1
        
    return output

def extract_label_images(filename_base, num_images, patch_size=IMG_PATCH_SIZE,
                         patch_stride=IMG_PATCH_STRIDE, img_base_name="satImage_%.3d"):
    #Extract labels from ground truth as label images
    gt_imgs = []
    for i in range(1, num_images+1):
        imageid = img_base_name % i
        image_filename = filename_base + imageid + ".png"
        if os.path.isfile(image_filename):
            print('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            gt_imgs.append(img)
        else:
            print('File ' + image_filename + ' does not exist')

    num_images = len(gt_imgs)
    print('Extracting patches...')
    gt_patches = [pixel_to_patch_labels(gt_imgs[i], patch_size, patch_stride) for i in range(num_images)]
    
    return gt_patches

def read_image_array(filename_base, num_images, img_base_name="satImage_%.3d"):
    #Load an array of images from the file system
    imgs = []
    for i in range(1, num_images+1):
        image_id = img_base_name % i
        image_filename = filename_base + image_id + ".png"
        if os.path.isfile(image_filename):
            print('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            imgs.append(img)
        else:
            print('File ' + image_filename + ' does not exist')

    return imgs
