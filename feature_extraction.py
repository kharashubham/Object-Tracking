import numpy as np
import cv2
import csv
from random import *

dest = 'Dataset/nonrigid_datasets/'
activities_img = ['skiing']

randBinList = lambda n: [randint(0,1) for b in range(1,n+1)]

def get_next_frame(frame_count):
    ok = False
    img = cv2.imread(str.format(dest + 'cliff-dive1/img%s.jpg' % str(frame_count).zfill(4)))
    if img is not None:
        frame_count += 1
        ok = True
    return ok, frame_count, img


def first_segmentation_mask(image, rectangle):
    mask = np.zeros(image.shape[:2], np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(image, mask, rectangle, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 1, 0).astype('uint8')
    image = image*mask2[:, :, np.newaxis]
    # plt.imshow(image), plt.colorbar(),plt.show()
    return image


def rectangle_read(file):
    f = open(file, 'r')
    rects = []
    for line in f:
        rect = line.split(' ')
        rect = [x for x in rect if x != '']
        rect = [x.replace('\n', '') for x in rect]
        rect = [int(float(x)) for x in rect]
        rects.append(rect)
    return rects


def get_first_bbox():
    rect = rectangle_read(dest + 'cliff-dive1/cliff-dive1_gt.txt')[0]
    return rect


def build_filters():
    """ returns a list of kernels in several orientations
    """
    filters = []
    ksizes = [3, 4]
    lambds = [15,16]
    for theta in np.arange(0, np.pi, np.pi / 8):
        for ksize in ksizes:
            for lambd in lambds:
                params = {'ksize':(ksize, ksize), 'sigma':1.0, 'theta':theta, 'lambd':lambd,
                          'gamma':0.02, 'psi':0, 'ktype':cv2.CV_32F}
                # kern = cv2.getGaborKernel(**params)
                # kern /= 1.5*kern.sum()
                # filters.append((kern, params))
    return filters


def distance_object_center(rectangle):
    y = (2 * rectangle[1] + rectangle[3])/2
    x = (2 * rectangle[0] + rectangle[2])/2
    dists = []
    for i in range(rectangle[1],rectangle[1]+rectangle[3]):
        for j in range(rectangle[0],rectangle[0]+rectangle[2]):
            dist = np.sqrt(np.power((i-x),2) + np.power((j-y),2))
            dists.append(dist)
    return dists


def process(img, filters):
    """ returns the img filtered by the filter list
    """
    accum = np.zeros_like(img)
    for kern, params in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum


def create_patches(image, rect, shape):
    # image = image[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
    patches = []
    for i in range(rect[1],rect[1]+rect[3], 8):
        for j in range(rect[0],rect[0]+rect[2], 8):
            patches.append(image[j-int(shape[1]/2):j+int(shape[1]/2), i-int(shape[0]/2):i+int(shape[0]/2)])
    return patches

# distances = distance_object_center(rects[0])
# print(distances[:5])


def feature_vector(patch):
    mean_hsv = []
    std_hsv = []
    mean = []
    std = []
    if len(patch) != 0:
        for i in range(0, patch.shape[2]):
            patch_hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            mean_hsv.append(np.mean(patch_hsv[:, :, i]))
            std_hsv.append((np.std(patch_hsv[:, :, i])))
            mean.append(np.mean(patch[:, :, i]))
            std.append((np.std(patch[:, :, i])))
        #filters = build_filters()
        #patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        #p = process(patch, filters)
        #p = np.array(p)
        #p = p.reshape(1,-1).tolist()
        features = np.append(mean_hsv, np.append(std_hsv, np.append(mean, std)))
        features.reshape(1,-1)
        return features
    else:
        return None


def get_dataset():
    features = []
    for activity in activities_img:
        print(activity)
        rects = rectangle_read(dest + activity + '/' + activity + '_gt.txt')
        frame_count = 0
        while True:
            img = cv2.imread(str.format(dest + activity + '/img%s.jpg' % str(frame_count).zfill(4)))
            print(str.format(dest + activity + '/img%s.jpg' % str(frame_count).zfill(4)))
            frame_count = frame_count + 1
            if img is None:
                print('patches done')
                break
            patches = create_patches(img, rects[frame_count-1], [8, 8])
            for patch in patches:
                if feature_vector(patch) is not None:
                    features.append(feature_vector(patch))
                else:
                    continue
    print('features done')
    # for activity in activities:
    #     rects = rectangle_read(dest + activity + '/' + activity + '_gt.txt')
    #     frame_count = 0
    #     while True:
    #         img = cv2.imread(str.format(dest + activity+'/%s.jpg' % str(frame_count).zfill(4)))
    #         patches = create_patches(img, rects[frame_count], [8, 8])
    #         if img is None:
    #             break
    #         for patch in patches:
    #             if feature_vector(patch) is not None:
    #                 features.append(feature_vector(patch))
    #             else:
    #                 continue

    target = randBinList(len(features))
    print(len(features))

    return target, features

t, fe = get_dataset()


def create_dataset(target, features):
    dataset_list = []
    for i in range(0, len(features)):
        if i%500 == 0:
            print('i', i)
        if i ==100000:
            break
        #feature_single = []
        feature = features[i]
        data = []
        #data.append(i+1)
        for j in range(0, len(feature)):
            f = feature[j]
            data.append(f)
        data.append(target[i])
        #feature_single.append(data)
        index = str.format('patch %s' % i)
        dataset_list.append({index: data})
    #print((dataset_list[0].values()[0]))
    print((dataset_list[1]))

    with open('dataset.csv', 'w') as fie:
        w = csv.writer(fie)
        for somedict in dataset_list:
            w.writerows(somedict.values())


create_dataset(t, fe)