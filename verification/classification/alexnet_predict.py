import os
import json

import cv2
import numpy as np

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import AlexNet


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize((360, 360)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # load image
    #img_path = "./input/test_OUT.png"
    #assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    #bnyImage = Image.open(img_path)
    #bnyImage = 255*np.array(bnyImage).astype('uint8')
    #img = cv2.cvtColor(np.array(bnyImage), cv2.COLOR_GRAY2BGR)
    #img = Image.fromarray(img)

    # load image
    img_path = "./input/test_OUT.png"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    bnyImage = Image.open(img_path)

    img = cv2.cvtColor(np.array(bnyImage), cv2.COLOR_GRAY2BGR)
    # one channel
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img_bin = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # find the left point
    a1 = contours[0]
    a2 = contours[1]
    left1 = tuple(a1[:, 0][a1[:, :, 0].argmin()])
    left2 = tuple(a2[:, 0][a2[:, :, 0].argmin()])
    if left1 < left2:
        leftMin = left1[0] - 25
    else:
        leftMin = left2[0] - 25
    if leftMin > 240:
        leftMin = 240
    img = img[0:360, leftMin:leftMin+360]
    #print(img.shape)
    
    img = 255 * np.array(img).astype('uint8')
    img = Image.fromarray(img)
    

    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './classification/male_class.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model = AlexNet(num_classes=3).to(device)

    # load model weights
    weights_path = "./models/maleAlexNet_2000.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path))

    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    class_indict[0] = 'mid-age'
    class_indict[1] = 'old-age'
    class_indict[2] = 'young'
    
    if class_indict[str(predict_cla)] == 'yM':
        print_res = "class: {}".format('young', predict[predict_cla].numpy())
    elif class_indict[str(predict_cla)] == 'oM':
        print_res = "class: {}".format('old-age')
    else:
        print_res = "class: {}".format('mid-age')
        
    #print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
    #                                             predict[predict_cla].numpy())
    plt.title(print_res)
    print(print_res)
    plt.show()
  
    

if __name__ == '__main__':
    main()
