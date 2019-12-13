from pdf2image import convert_from_path

def cvtpdf2img(filename):
    pages = convert_from_path(filename+'.PDF', 500)

    for i, page in enumerate(pages):
        page.save('data/out_' + str(filename) + '_' + str(i) + '.jpg', "JPEG")

    return i


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
import json
from yolo.backend.utils.box import draw_scaled_boxes
import yolo
from yolo.frontend import create_yolo
import traceback

yolo_detector = create_yolo("ResNet50", ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ":"], 416)
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ":"]
DEFAULT_WEIGHT_FILE = "weights/weights.h5"
yolo_detector.load_weights(DEFAULT_WEIGHT_FILE)

model = Net()
alpha_classes = (["C","I","W","blank"])


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


def predict_time(img, THRESHOLD=0.2):
    boxes, probs = yolo_detector.predict(img, THRESHOLD)

    image, scaled_boxes = draw_scaled_boxes(img,
                                boxes,
                                probs,
                                ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ":"])
    
    detected = []
    for prob in probs:
        detected.append(classes[np.argmax(prob)])
    
    final = []
    for i in range(len(scaled_boxes)):
        dict = {}
        dict['x'] = scaled_boxes[i][0]
        dict['w'] = scaled_boxes[i][1]
        dict['y'] = scaled_boxes[i][2]
        dict['h'] = scaled_boxes[i][3]
        dict['class'] = detected[i]
        
        final.append(dict)

    final=sorted(final, key=lambda x:x['x'])
    detected =  [sub['class'] for sub in final]
    detected = ''.join(detected)
    #print('Detected: ', detected)

    return detected

model = Net()
alpha_classes = (["C","I","W","blank"])

def make_straight(img):

    ### Make straight
    print('.')
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



    #cv2.imshow('time', img_psp)
    #cv2.waitKey(0)
    #cv2.imwrite('tmp.png', img_psp)

    return img_psp
    #####################
    # END Make straight #
    #####################


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
#   time   #
############

def time(img):
    print('.')
    H = img.shape[0]
    W = img.shape[1]
    times = []
    for i in range(0, H, 74):
        input = img[i:i+74, :]
        #cv2.imshow('input',input)
        #cv2.waitKey(0)
        detected = predict_time(input)
        times.append(detected)
    return times

############
# Alphabet #
############
def alphabet(img, prev, step):
    print('.')
    model = Net()
    param = torch.load('weights/alphabet3.pth', map_location='cpu') #delete map_location if you use cuda #model__Feb_5.pth
    model.load_state_dict(param)

    alphabets = []
    H = img.shape[0]
    W = img.shape[1]
    for i in range(0, H, 74):
        im = img[i+5:i+74-5, (int(W/2)-35):(int(W/2)+35)]
        # Convert to grayscale and apply Gaussian filtering
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        #im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
    
        # Threshold the image
        ret, input_ = cv2.threshold(im_gray, 170, 255, cv2.THRESH_BINARY)
        
        input_ = cv2.resize(input_, (32,32))
        #cv2.imshow("image", input_)
        #cv2.waitKey(0)
                                 
        input_ = Image.fromarray(input_)
        pred = predict_alphabet(input_, model)
        #print(pred)
        cv2.putText(im, str(pred), (10, 20),cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 0), 2)

        if pred == 'blank':
            alphabets.append('')
        else:
            alphabets.append(pred[0])

            
        #cv2.imshow("image", im)
        #cv2.waitKey(0)

    return alphabets
################
# END ALPHABET #
################

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
        for data in lists:
            print(data)
            wr.writerow(data)



if __name__ == '__main__':
    args = sys.argv
    if args[1]:
        file = args[1].split('.PDF')[0]
        print('get arg', file)
    else:
        print("Don't get args loading default pdf ...")
        file = 'null1No.35_39.PDF'
    cnt = cvtpdf2img(file)
    starts = []
    ends = []
    bhvs = []
    
    #for imfile in glob.glob('data/out_'+file+'*.jpg'):
    for i in range(cnt+1):
        try:
            imfile = 'data/out_' + file + '_' + str(i) + '.jpg'
            print("loading " + imfile + '...')
            img = cv2.imread(imfile)
            img  = make_straight(img)

            img_start = img.copy()
            img_start = img_start[55:, 220:440]
            img_start = cv2.fastNlMeansDenoisingColored(img_start,None,10,10,7,21)
            #cv2.imshow('start',img_start)
            #cv2.waitKey(0)
            
            img_end = img.copy()
            img_end = img_end[55:, 440:660]
            img_end = cv2.fastNlMeansDenoisingColored(img_end,None,10,10,7,21)
            #plt.imshow(img_end)
            #plt.title('end')
            #plt.show()
            
            img_alphabet = img.copy()
            img_alphabet = img_alphabet[55:, 880:1100]
            img_alphabet = cv2.fastNlMeansDenoisingColored(img_alphabet,None,10,10,7,21)
            #plt.imshow(img_alphabet)
            #plt.title('alpaebat')
            #plt.show()
            #cv2.imshow('alpha',img_alphabet)
            #cv2.waitKey(0)
            #img = img[55:, 220:1110]
            
            start = time(img_start)
            for val in start:
                starts.append(val)
            
            end = time(img_end)
            for val in end:
                ends.append(val)
               
            prev = 0
            step = int(img.shape[0]/31)
        
            classes_alpha = ('C', 'I','W','blank')
            
            episodes =[]
            i = 0
            alphabets = alphabet(img_alphabet, prev, step)

            for bhv in alphabets:
                bhvs.append(bhv)

        except:
            traceback.print_exc()
            
#print('st: ',starts[32])
#print('en: ',ends[32]))
#print(bhvs[32])
output = [['start', 'end', 'episode']]

for i in range(len(starts)):
    print(i)
    tmp = [starts[i], ends[i], bhvs[i]]
    output.append(tmp)

print(output)
export2csv(file, output)
