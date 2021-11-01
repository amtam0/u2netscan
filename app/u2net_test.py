import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
# import torch.optim as optim
import numpy as np
from PIL import Image
import glob
# from data_loader import RescaleT
# from data_loader import ToTensor
# from data_loader import ToTensorLab
# from data_loader import SalObjDataset
from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB

model_name = "u2netp"
if model_name=="u2netp":
    model_dir = "saved_models/u2netp/u2netp.pth"
    net = U2NETP(3,1)
else:
    model_dir = "saved_models/u2net/u2net.pth"
    net = U2NET(3,1)

if torch.cuda.is_available():
    print("CUDA")
    net.load_state_dict(torch.load(model_dir)) #ATA
    net.cuda()
else:
    net.load_state_dict(torch.load(model_dir,map_location='cpu')) #ATA 
net.eval()

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def save_output(image_name,pred,d_dir):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

    pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir+imidx+'.png')

def ToTensorLab(image=None,imidx=None,label=None,flag=0):
    
    tmpLbl = np.zeros(label.shape)

    if(np.max(label)<1e-6):
        label = label
    else:
        label = label/np.max(label)
        
    # change the color space
    if flag == 2: # with rgb and Lab colors
        tmpImg = np.zeros((image.shape[0],image.shape[1],6))
        tmpImgt = np.zeros((image.shape[0],image.shape[1],3))
        if image.shape[2]==1:
            tmpImgt[:,:,0] = image[:,:,0]
            tmpImgt[:,:,1] = image[:,:,0]
            tmpImgt[:,:,2] = image[:,:,0]
        else:
            tmpImgt = image
        tmpImgtl = color.rgb2lab(tmpImgt)

        # nomalize image to range [0,1]
        tmpImg[:,:,0] = (tmpImgt[:,:,0]-np.min(tmpImgt[:,:,0]))/(np.max(tmpImgt[:,:,0])-np.min(tmpImgt[:,:,0]))
        tmpImg[:,:,1] = (tmpImgt[:,:,1]-np.min(tmpImgt[:,:,1]))/(np.max(tmpImgt[:,:,1])-np.min(tmpImgt[:,:,1]))
        tmpImg[:,:,2] = (tmpImgt[:,:,2]-np.min(tmpImgt[:,:,2]))/(np.max(tmpImgt[:,:,2])-np.min(tmpImgt[:,:,2]))
        tmpImg[:,:,3] = (tmpImgtl[:,:,0]-np.min(tmpImgtl[:,:,0]))/(np.max(tmpImgtl[:,:,0])-np.min(tmpImgtl[:,:,0]))
        tmpImg[:,:,4] = (tmpImgtl[:,:,1]-np.min(tmpImgtl[:,:,1]))/(np.max(tmpImgtl[:,:,1])-np.min(tmpImgtl[:,:,1]))
        tmpImg[:,:,5] = (tmpImgtl[:,:,2]-np.min(tmpImgtl[:,:,2]))/(np.max(tmpImgtl[:,:,2])-np.min(tmpImgtl[:,:,2]))

        # tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

        tmpImg[:,:,0] = (tmpImg[:,:,0]-np.mean(tmpImg[:,:,0]))/np.std(tmpImg[:,:,0])
        tmpImg[:,:,1] = (tmpImg[:,:,1]-np.mean(tmpImg[:,:,1]))/np.std(tmpImg[:,:,1])
        tmpImg[:,:,2] = (tmpImg[:,:,2]-np.mean(tmpImg[:,:,2]))/np.std(tmpImg[:,:,2])
        tmpImg[:,:,3] = (tmpImg[:,:,3]-np.mean(tmpImg[:,:,3]))/np.std(tmpImg[:,:,3])
        tmpImg[:,:,4] = (tmpImg[:,:,4]-np.mean(tmpImg[:,:,4]))/np.std(tmpImg[:,:,4])
        tmpImg[:,:,5] = (tmpImg[:,:,5]-np.mean(tmpImg[:,:,5]))/np.std(tmpImg[:,:,5])

    elif flag == 1: #with Lab color
        tmpImg = np.zeros((image.shape[0],image.shape[1],3))

        if image.shape[2]==1:
            tmpImg[:,:,0] = image[:,:,0]
            tmpImg[:,:,1] = image[:,:,0]
            tmpImg[:,:,2] = image[:,:,0]
        else:
            tmpImg = image

        tmpImg = color.rgb2lab(tmpImg)

        # tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

        tmpImg[:,:,0] = (tmpImg[:,:,0]-np.min(tmpImg[:,:,0]))/(np.max(tmpImg[:,:,0])-np.min(tmpImg[:,:,0]))
        tmpImg[:,:,1] = (tmpImg[:,:,1]-np.min(tmpImg[:,:,1]))/(np.max(tmpImg[:,:,1])-np.min(tmpImg[:,:,1]))
        tmpImg[:,:,2] = (tmpImg[:,:,2]-np.min(tmpImg[:,:,2]))/(np.max(tmpImg[:,:,2])-np.min(tmpImg[:,:,2]))

        tmpImg[:,:,0] = (tmpImg[:,:,0]-np.mean(tmpImg[:,:,0]))/np.std(tmpImg[:,:,0])
        tmpImg[:,:,1] = (tmpImg[:,:,1]-np.mean(tmpImg[:,:,1]))/np.std(tmpImg[:,:,1])
        tmpImg[:,:,2] = (tmpImg[:,:,2]-np.mean(tmpImg[:,:,2]))/np.std(tmpImg[:,:,2])

    else: # with rgb color
        tmpImg = np.zeros((image.shape[0],image.shape[1],3))
        image = image/np.max(image)
        if image.shape[2]==1:
            tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
            tmpImg[:,:,1] = (image[:,:,0]-0.485)/0.229
            tmpImg[:,:,2] = (image[:,:,0]-0.485)/0.229
        else:
            tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
            tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
            tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225

    tmpLbl[:,:,0] = label[:,:,0]

    # change the r,g,b to b,r,g from [0,255] to [0,1]
    #transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
    tmpImg = tmpImg.transpose((2, 0, 1))
    tmpLbl = label.transpose((2, 0, 1))
    
    return {'imidx':torch.from_numpy(imidx), 'image': torch.from_numpy(tmpImg), 'label': torch.from_numpy(tmpLbl)}

def RescaleT(image=None,imidx=None,label=None,output_size=320):
    
    h, w = image.shape[:2]

    if isinstance(output_size,int):
        if h > w:
            new_h, new_w = output_size*h/w,output_size
        else:
            new_h, new_w = output_size,output_size*w/h
    else:
        new_h, new_w = output_size

    new_h, new_w = int(new_h), int(new_w)

    # #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
    # img = transform.resize(image,(new_h,new_w),mode='constant')
    # lbl = transform.resize(label,(new_h,new_w),mode='constant', order=0, preserve_range=True)

    img = transform.resize(image,(output_size,output_size),mode='constant')
    lbl = transform.resize(label,(output_size,output_size),mode='constant', order=0, preserve_range=True)
    
    return {'imidx':imidx, 'image':img,'label':lbl}
    
def SalObjDataset(img_name_list=None,lbl_name_list=None,idx=0):
    
    image = io.imread(img_name_list[idx])
    imname = img_name_list[idx]
    imidx = np.array([idx])
    
    if(0==len(lbl_name_list)):
        print("0==len(lbl_name_list)")
        label_3 = np.zeros(image.shape)
    else:
        label_3 = io.imread(lbl_name_list[idx])

    label = np.zeros(label_3.shape[0:2])
    if(3==len(label_3.shape)):
        print("3==len(label_3.shape)")
        label = label_3[:,:,0]
    elif(2==len(label_3.shape)):
        print("2==len(label_3.shape)")
        label = label_3

    if(3==len(image.shape) and 2==len(label.shape)):
        label = label[:,:,np.newaxis]
        print("3==len(image.shape) and 2==len(label.shape)")
    elif(2==len(image.shape) and 2==len(label.shape)):
        image = image[:,:,np.newaxis]
        label = label[:,:,np.newaxis]
        print("2==len(image.shape) and 2==len(label.shape)")
    
    return {'imidx':imidx, 'image':image, 'label':label}

def main(model_name="u2netp",image_dir = "/tmp/in_data/",prediction_dir = "/tmp/out_data/"):

    # --------- 1. get image path and name ---------
    
    if model_name=="u2netp":
        model_dir = "saved_models/u2netp/u2netp.pth"
    else:
        model_dir = "saved_models/u2net/u2net.pth"

    img_name_list = glob.glob(image_dir + os.sep + '*')
    print(img_name_list)

    # --------- 2. dataloader ---------
    #1. dataloader
#     test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
#                                         lbl_name_list = [],
#                                         transform=transforms.Compose([RescaleT(320),
#                                                                       ToTensorLab(flag=0)])
#                                         )
#     test_salobj_dataloader = DataLoader(test_salobj_dataset,
#                                         batch_size=1,
#                                         shuffle=False,
#                                         num_workers=1)
    
    ###ATA####
    sample = SalObjDataset(img_name_list=img_name_list,lbl_name_list=[],idx=0)
    imidx,image,label = sample["imidx"],sample["image"],sample["label"]
    sample = RescaleT(image=image,imidx=imidx,label=label,output_size=320)
    imidx,image,label = sample["imidx"],sample["image"],sample["label"]
    data_test = ToTensorLab(image=image,imidx=imidx,label=label,flag=0)
    
    ###ATA####
    
    # --------- 3. model define ---------
    """if(model_name=='u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3,1)
    elif(model_name=='u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3,1)
    if torch.cuda.is_available():
        print("CUDA")
        net.load_state_dict(torch.load(model_dir)) #ATA ,map_location='cpu'
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir,map_location='cpu')) #ATA 
    net.eval()
    """
    print("MODEL LOADED OK")

    # --------- 4. inference for each image ---------
#     for i_test, data_test in enumerate(test_salobj_dataloader):

    print("inferencing:",img_name_list[imidx[0]].split(os.sep)[-1])

    inputs_test = data_test['image']
    inputs_test = inputs_test.type(torch.FloatTensor)

    if torch.cuda.is_available():
        inputs_test = Variable(inputs_test.cuda())
    else:
        inputs_test = Variable(inputs_test)

    d1,d2,d3,d4,d5,d6,d7= net(inputs_test.unsqueeze_(0)) #ATA add .unsqueeze_(0)

    # normalization
    pred = d1[:,0,:,:]
    pred = normPRED(pred)

    # save results to prediction_dir folder
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir, exist_ok=True)
    
    save_output(img_name_list[imidx[0]],pred,prediction_dir)

    del d1,d2,d3,d4,d5,d6,d7
    
# if __name__ == "__main__":
#     main()
