import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def process_image(image):
    """ Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    """
    # TODO: Process a PIL image for use in a PyTorch model

    crop_size = 224
    new_width = 256
    # ImageNet mean and std
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # 1. Resize with shortest side 256 and maintain aspect ratio
    width, height = image.size
    orig_ar = (width / height)
    new_height = (new_width / orig_ar)
    image.thumbnail((new_width, new_height), Image.ANTIALIAS)

    # 2. center crop
    left = int((new_width - crop_size) / 2)
    top = int((new_height - crop_size) / 2)
    right = int((new_width + crop_size) / 2)
    bottom = int((new_height + crop_size) / 2)
    image = image.crop((left, top, right, bottom))

    # 3. PIL to nparray - 0-255
    np_image = np.array(image)
    # scale : 0 to 1 range
    np_image = np_image / 255.0

    # 4. Normalize the image
    np_image = (np_image - mean) / std

    # 5. re-order color channels
    out_image = np.transpose(np_image, (2, 0, 1))

    return out_image


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo pre-processing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax


def view_classify(image_torch, top_prob,  top_class, topk, cat_to_name=[]):

    if cat_to_name:
        class_names = [cat_to_name[item] for item in top_class]
    else:
        class_names = top_class

    print(class_names)

    fig, (ax1, ax2) = plt.subplots(figsize=(6, 15), ncols=2)
    ax1 = imshow(image_torch, ax=ax1)
    ax1.axis('off')

    ax2.barh(np.arange(topk), list(reversed(top_prob)))
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(topk))
    ax2.set_yticklabels(reversed(class_names), size='small')
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 0.4)

    plt.tight_layout()
