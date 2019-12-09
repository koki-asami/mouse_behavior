from pdf2image import convert_from_path

def cvtpdf2img(filename, num):
    pages = convert_from_path(filename, 500)

    for i, page in enumerate(pages):
        page.save('data/out_' + str(num) + '_' + str(i) + '.jpg', "JPEG")


import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from model import Net
import glob
import torch
from torchvision import transforms
import csv
import random
import os
import sys

model = Net()

alpha_classes = (["C","I","W","blank"])

def make_straight(img):

    ### Make straight
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    
    contours, hierarchy = cv2.findContours(thresh , cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )

    menseki=[ ]

    for i in range(0, len(contours)): 
        menseki.append([contours[i],cv2.contourArea(contours[i])])

    menseki.sort(key=lambda x: x[1], reverse=True)

    epsilon = 0.1*cv2.arcLength(menseki[1][0],True)
    approx = cv2.approxPolyDP(menseki[1][0],epsilon,True)

    cv2.drawContours(img, approx, -1,(0, 0, 255),10)
    #cv2.imwrite("result.png",img)
    approx=approx.tolist()

    left = sorted(approx,key=lambda x:x[0]) [:2]
    right = sorted(approx,key=lambda x:x[0]) [2:]

 

    left_down= sorted(left,key=lambda x:x[0][1]) [0]
    left_up= sorted(left,key=lambda x:x[0][1]) [1]

 

    right_down= sorted(right,key=lambda x:x[0][1]) [0]
    right_up= sorted(right,key=lambda x:x[0][1]) [1]

    perspective1 = np.float32([left_down,right_down,right_up,left_up])
    perspective2 = np.float32([[0, 0],[1654, 0],[1654, 2340],[0, 2340]])

 

    psp_matrix = cv2.getPerspectiveTransform(perspective1,perspective2)
    img_psp = cv2.warpPerspective(img, psp_matrix,(1654,2340))

 

    #cv2.imwrite("image_modified.png",img_psp)
    #cv2.imshow('output', img_psp)
    #cv2.waitKey(0)

    return img_psp
    #####################
    # END Make straight #
    #####################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((32,32)),                           
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


def predict_alphabet(image, model_):
    model.eval()
    image_tensor = transform(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = image_tensor.to(device)
    output = model_(input)
    _, pred = torch.max(output, 1)
    return alpha_classes[pred.item()]





def sort(list_, idx):
    sorted_ = []
    while len(list_):
        min_, idx = find_min(list_, idx)
        for val in list_:
            if val[0] == min_:
                sorted_.append(val)
                list_.remove(val)
    return sorted_

def find_min(list_, idx):
    tmp = 9999
    for i in range(len(list_)):
        if list_[i][idx] < tmp:
            tmp = list_[i][idx]
            idx_ = idx
    return int(tmp), idx_


############
# Alphabet #
############
def alphabet(img, prev, step):
    model = Net()
    param = torch.load('weights/alphabet3.pth', map_location='cpu') #delete map_location if you use cuda #model__Feb_5.pth
    model.load_state_dict(param)
    
    
    i = 0
    j = 0
    while(prev + step <= img.shape[0]):
        episode = []
        
        if i > 0:
            #print(prev+5, prev+step)
            im = img[prev+6:prev+step-2,690:855]
            #prev += 10
        else:
            #print(0, step)
            im = img[5:step, 690:855] #250:430 <- end
            #cv2.imshow("input", im)
            #cv2.waitKey(0)
            

        # Convert to grayscale and apply Gaussian filtering
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
    
        # Threshold the image
        ret, im_th_INV = cv2.threshold(im_gray, 170, 255, cv2.THRESH_BINARY)
        #ret, im_th = cv2.threshold(im_gray, 125, 255, cv2.THRESH_BINARY)
        im_th_INV = cv2.copyMakeBorder(im_th_INV,10,10,10,10,cv2.BORDER_CONSTANT,value=[255,255,255])#value=[255,255,255])
        #cv2.imshow("Resulting Image with Rectangular ROIs", im_th_INV)
        #cv2.waitKey(0)
        
        # Find contours in the image
        ctrs, hier = cv2.findContours(im_th_INV.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get rectangles contains each contour
        rects = [cv2.boundingRect(ctr) for ctr in ctrs if cv2.contourArea(ctr) > 50 ]#and cv2.contourArea(ctr) < 8000]

                
        im = cv2.cvtColor(im_th_INV, cv2.COLOR_GRAY2BGR)
        for i, rect in enumerate(rects):
            
            cv2.rectangle(im, (rect[0]+40, rect[1]+10), (rect[0] + rect[2]-40, rect[1] + rect[3]-10), (0, 255, 0), 1)
            
            input_ = im_th_INV[rect[1]+10:rect[1]+rect[3]-10, rect[0]+40:rect[0]+rect[2]-40]
            input_ = cv2.copyMakeBorder(input_,0,0,10,10,cv2.BORDER_CONSTANT,value=[255,255,255])#value=[255,255,255])
            input_ = cv2.resize(input_, (32,32))
            #cv2.imshow("image", input_)
            #cv2.waitKey(0)
            #cv2.imwrite("raw/" +"alphabet" + str(random.randint(0,10000)) + "11_input" + str(j) + ".png", input_)
                     
            input_ = Image.fromarray(input_)
            pred = predict_alphabet(input_, model)
            #print(pred)
            cv2.putText(im, str(pred), (10, 20),cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 0), 2)

            if pred == 'blank':
                episode.append('')
            else:
                episode.append(pred[0])

            


        episodes.append(episode[0])

        prev += step
        i += 1
        j += 1
        #cv2.imshow("image", im)
        #cv2.waitKey(0)

    return episodes
        ################
        # END ALPHABET #
        ################
        ######

def touch(path):
    if os.path.isfile(path):
        pass
    else:
        with open(path, "w", encoding="UTF-8") as f:
            pass

def export2csv(imfile, lists):
    
    filename = ('csv/'+ imfile + '.csv')
    touch(filename)
    with open(filename, 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(lists)



if __name__ == '__main__':
    args = sys.argv
    if args[0]:
        file = args[1].split('.')[0]
        print('get arg', file)
    else:
        file = 'pernull2 No.1-14.PDF'
    #cvtpdf2img('pernull2 No.1-14.PDF', 7)
    bhvs = []
    cnt = 0
    for imfile in glob.glob('data/out*.jpg'):
        try:
            print("loading " + imfile + '...')
            img = cv2.imread(imfile)
            img  = make_straight(img)
            img = img[55:, 220:1110]
            
            img = cv2.resize(img, (888,2263))
            prev = 0
            step = int(img.shape[0]/31)
        
            classes_alpha = ('C', 'I','W','blank')
            
            episodes =[]
            i = 0
            alphabets = alphabet(img, prev, step)

            for bhv in alphabets:
                bhvs.append(bhv)

        except:
            pass

print(bhvs)
export2csv(file, bhvs)
