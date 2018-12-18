import numpy as np
from skimage.transform import resize as imresize
from imageio import imread as imread
import matplotlib.pyplot as plt


def draw_class_activation_map(img, class_activation_map, img_alpha=0.6, size=None):
    # resize input images
    if size is not None:
        r = size / np.minimum(img.shape[0], img.shape[1])
        img = imresize(
            img, output_shape=[int(r * img.shape[0] + 0.5), int(r * img.shape[1] + 0.5)], preserve_range=True)

    class_activation_map = imresize(class_activation_map, output_shape=[img.shape[0], img.shape[1]])

    # create rgb overlay
    cm = plt.get_cmap('jet')
    cam_ovlr = cm(class_activation_map)

    # normalize to 0..1 and convert to grayscale
    img = img / 255.0
    img_gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

    # create heatmap composite
    cam_heatmap = img_alpha * np.expand_dims(img_gray, axis=-1) + (1 - img_alpha) * cam_ovlr[:, :, 0:3]

    # visualize
    plt.imshow(cam_heatmap)
    plt.show()


def calc_class_activation_map(network_dict, output_dict, class_idx, partition_idx=-1):
    # get weights and activations of specified class and partition
    class_activation_weights = network_dict['activation_weights'][partition_idx][0, 0, :, class_idx]
    activations = output_dict['activations']

    # get dimensions
    num_crops, h, w, num_features = activations.shape
    img_size = output_dict['image_bboxes'][0][-1]
    r = h / img_size

    # create output variables
    cam = np.zeros(shape=[
        int(output_dict['image_bboxes'][-1][0] * r + 0.5) +
        w, int(output_dict['image_bboxes'][-1][1] * r + 0.5) + h
    ])

    num_activations = np.zeros(shape=[
        int(output_dict['image_bboxes'][-1][0] * r + 0.5) +
        w, int(output_dict['image_bboxes'][-1][1] * r + 0.5) + h
    ])

    # generate class activation map for each image crop
    for crop_idx in range(num_crops):
        # get activation map of current crop
        crop_activations = activations[crop_idx, :, :, :]
        crop_activation_map = class_activation_weights.dot(crop_activations.reshape((num_features, h * w)))
        crop_activation_map = crop_activation_map.reshape(h, w)

        # translate bbox coordinates from original image size to feature size
        feature_bbox = []
        for entry in output_dict['image_bboxes'][crop_idx]:
            feature_bbox.append(int(entry * r + 0.5))

        # store class activation map of the crop
        cam[feature_bbox[0]:feature_bbox[0] + w, feature_bbox[1]:feature_bbox[1] + h] += crop_activation_map
        num_activations[feature_bbox[0]:feature_bbox[0] + w, feature_bbox[1]:feature_bbox[1] + h] += 1

    # NOTE: prevent division by 0, if the whole image is not covered with all crops [max_dim > 3 * min_dim]
    num_activations[num_activations == 0] = 1

    # normalize class activation map
    cam /= num_activations
    cam -= np.min(cam)
    cam /= np.max(cam)
    cam = np.asarray(cam * 255 + 0.5, dtype=np.uint8)

    return cam
