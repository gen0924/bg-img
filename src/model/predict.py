import os
import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms
from torch.autograd import Variable
from skimage import img_as_bool, img_as_uint, morphology
import cv2
import src.model.transforms as ext_transforms
from src.model.enet import ENet
from collections import OrderedDict
import time
from src.func.img_handler import show_img

Height = 360
Width = 480

color_encoding = OrderedDict([('unlabeled', (0, 0, 0)), ('screen', (255, 255, 255))])


def load_model():

    num_classes = len(color_encoding)
    model = ENet(num_classes)
    currentpath = os.path.dirname(__file__)
    save_dir = currentpath + '/'
    name = 'ENet_best_65'
    model = load_checkpoint(model, save_dir, name)

    return model


def load_checkpoint(model, folder_dir, filename):
    assert os.path.isdir(
        folder_dir), "The directory \"{0}\" doesn't exist.".format(folder_dir)

    # Create folder to save model and information
    model_path = os.path.join(folder_dir, filename)
    assert os.path.isfile(
        model_path), "The model file \"{0}\" doesn't exist.".format(filename)

    # Load the stored model parameters to the model instance
    start = time.time()
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    end = time.time()
    print('Load model Running Time : %s Sec ' % (end - start))

    return model


def batch_transform(batch, transform):
    transf_slices = [transform(tensor) for tensor in torch.unbind(batch)]

    return torch.stack(transf_slices)


def predict(model, src, class_encoding):
    image_transform = transforms.Compose([transforms.Resize((Height, Width)),
                                          transforms.ToTensor()])
    img = image_transform(src).unsqueeze(0)
    images = Variable(img)
    start = time.time()
    predictions = model(images)

    end = time.time()
    print('Segment Time : %s Sec ' % (end - start))
    # Predictions is one-hot encoded with "num_classes" channels.
    # Convert it to a single int using the indices where the maximum (1) occurs
    _, predictions = torch.max(predictions.data, 1)

    label_to_rgb = transforms.Compose([
        ext_transforms.LongTensorToRGBPIL(class_encoding),
        transforms.ToTensor()
    ])
    labels = batch_transform(predictions.cpu(), label_to_rgb)

    labels = torchvision.utils.make_grid(labels).numpy()
    labels = np.transpose(labels, (1, 2, 0))
    labels = cv2.cvtColor(np.asarray(labels), cv2.COLOR_RGB2GRAY)
    labels = (labels * 255).astype(np.uint8)
    # print(labels.dtype, type(labels), np.max(labels))
    # show_img(labels, 0)
    ret, labels = cv2.threshold(labels, 5, 255, cv2.THRESH_BINARY)
    # print(labels.dtype, type(labels), np.max(labels))
    # show_img(labels, 0)

    # 形态学操作-最大轮廓
    imgbool = img_as_bool(labels)
    img = morphology.remove_small_objects(imgbool, min_size=5000, connectivity=1, in_place=True)
    dst = img_as_uint(img)
    # show_img(dst, 0)
    # 图像膨胀
    kernel = np.ones((10, 10), np.uint8)
    dst = cv2.dilate(dst, kernel)
    dst = cv2.resize(dst, (768, 768), interpolation=cv2.INTER_AREA)

    return dst




