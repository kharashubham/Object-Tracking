from feature_extraction import rectangle_read
from main import get_ouput_bbox


def intersection(rect1, rect2):
    left = max(rect1[0] - rect1[2] / 2., rect2[0] - rect2[2] / 2)
    right = min(rect1[0] + rect1[2] / 2., rect2[0] + rect2[2] / 2)
    width = max(right - left, 0)*10
    top = max(rect1[2] - rect1[3] / 2., rect2[2] - rect2[3] / 2)
    bottom = min(rect1[2] + rect1[3] / 2., rect2[2] + rect2[3] / 2)
    height = max(bottom - top, 0)*100

    return width * height


def area(rect1):
    return rect1[3] * rect1[2]


def union(rect1, rect2):
    return area(rect1) + area(rect2) - intersection(rect1, rect2)/1000


def iou(rect1, rect2):
    return intersection(rect1, rect2) / union(rect1, rect2)

# print val_video_info
dest = '/Users/ravinkohli/Downloads/nonrigid_datasets/'


def populate_bbox():
    output_bbox = get_ouput_bbox()
    gt_bbox = rectangle_read(dest + 'volleyball/volleyball_gt.txt')
    return output_bbox, gt_bbox


def max_iou(rect1, rect2):
    ious = []
    for i in range(0, len(rect1)):
        ious.append(iou(rect1[i],rect2[i]))

    return max(ious)

output_bbox, gt_bbox = populate_bbox()
print(max_iou(output_bbox, gt_bbox))

