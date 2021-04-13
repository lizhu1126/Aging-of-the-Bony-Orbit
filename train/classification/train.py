import os
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from model import AlexNet
from tensorboardX import SummaryWriter


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("using {} device.".format(device))

    tbwriter=SummaryWriter(log_dir="./logs")

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(360),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize(360, 360),  # cannot 360, must (360,360)
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}


    data_root = os.path.abspath(os.path.join(os.getcwd(), "./DATA"))  # get data root path
    image_path = os.path.join(data_root, "male")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=2)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 8
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=8, shuffle=True,
                                                  num_workers=nw)

    print("using {} images for training, {} images fot validation.".format(train_num,
                                                       val_num))


    if os.path.exists("./log360.pth"):
        net=AlexNet()
        #net.load_state_dict(torch.load("./log360.pth", map_location='cuda:2'))
        net=torch.load("./log360.pth", 'cpu')
        print("continue training")
    else:
        net = AlexNet(num_classes=3, init_weights=True)
        net.to(device)
        print("start training anew")

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.98)

    epochs = 2000
    save_path = './AlexNet.pth'
    best_acc = 0.0
    train_steps = len(train_loader)

    #json_path = './class_indices.json'
    #json_file = open(json_path, "r")
    #class_indict = json.load(json_file)
    #model = AlexNet(num_classed=6).to(device)
        
    trainLOSS = []  #save loss
    testLOSS = []  #save loss
    valACC = []     #save val acc

    for epoch in range(epochs):
        scheduler.step()
        print('LR:{}'.format(scheduler.get_lr()[0]))
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, colour='green')
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num

        tbwriter.add_scalar('train/loss', running_loss/train_steps, epoch)
        tbwriter.add_scalar('val/acc', val_accurate, epoch)

        trainLOSS.append(running_loss/train_steps)
        valACC.append(val_accurate)

        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))
        print(' ')

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
        
        #predict
        #weights_path="./AlexNet.pth"
        #model.load_state_dict(torch.load(weights_path))

        #model.eval()
        #with torch.no_grad():
        #    putput = torch.squeeze(model(img.to(device))).cpu()
        #    predict = torch.softmax(output, dim=0)
        #    predict_cla = torch.argmax(predict.numpy)


    npLOSS=np.array(trainLOSS)
    npVALACC=np.array(valACC)
    np.save('./save/loss_epoch_{}'.format(epoch), npLOSS)
    np.save('./save/valacc_epoch_{}'.format(epoch), npVALACC)

    print('Finished Training')





if __name__ == '__main__':
    main()
