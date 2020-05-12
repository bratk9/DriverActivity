import cv2
import math
import argparse
import time

def highlightFace(net, frame, conf_threshold=0.5):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes = []
    faceCenters = []
    faceNum = 0
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            faceCenters.append( [(x1 + x2)//2, (y1 + y2)//2] )
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,0,255), 2, 8)
            cv2.circle(frameOpencvDnn, (faceCenters[i][0], faceCenters[i][1]), 2, (255,255,0), -1)
            faceNum += 1
    return frameOpencvDnn, faceBoxes, faceNum, faceCenters

#given a face center this function returns 0 if it's in driver seat , 1 if in passenger seat , 2 for back seats
def findRegionForFace(facePoint):
    x = facePoint[0]
    y = facePoint[1]
    
    if x < x2 and y > y1:
        return 0
        
    elif x > x3  and y > y3:
        return 1
    
    return 2
    

# parser=argparse.ArgumentParser()
# parser.add_argument('--image')

# args=parser.parse_args()

faceProto="opencv_face_detector.pbtxt"
faceModel="opencv_face_detector_uint8.pb"
ageProto="age_deploy.prototxt"
ageModel="age_net.caffemodel"
genderProto="gender_deploy.prototxt"
genderModel="gender_net.caffemodel"

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']

car_passenger_limit = 4
font = cv2.FONT_HERSHEY_SIMPLEX
line = cv2.LINE_AA

# reg_point[0] is rectangular region for driver seat: '[x1,y1,x2,y2] format' and reg_point[1] for passenger seat
reg_point = [ [0, 185, 250, 480], [390, 185, 640, 480] ]

x1 = reg_point[0][0]
y1 = reg_point[0][1]
x2 = reg_point[0][2]
y2 = reg_point[0][3]
x3 = reg_point[1][0]
y3 = reg_point[1][1]
x4 = reg_point[1][2]
y4 = reg_point[1][3]

faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)

video=cv2.VideoCapture(0)
padding=20

hasFrame, frame = video.read()
height, width, layers = frame.shape
size = (width, height)
# out = cv2.VideoWriter('video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 7, size)


while True:
    hasFrame,frame=video.read()
    frame = cv2.resize(frame, (640, 480))
    if cv2.waitKey(1) & 255 == 27:
        break

        
    resultImg, faceBoxes, faceNum, faceCenters = highlightFace(faceNet,frame)
    if not faceBoxes:
        print("No face detected")
    
    print(f'No. of faces: {faceNum}')    
    childNum = 0
    
    childInFront = False
    passengerExtra = False
    for i, faceBox in enumerate(faceBoxes):
        face=frame[max(0,faceBox[1]-padding):
                   min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                   :min(faceBox[2]+padding, frame.shape[1]-1)]

        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds=genderNet.forward()
        gender=genderList[genderPreds[0].argmax()]
        print(f'Face No. : {i}')
        print(f'Gender: {gender}')

        ageNet.setInput(blob)
        agePreds=ageNet.forward()
        age=ageList[agePreds[0].argmax()]
        print(f'Age: {age[1:-1]} years')
        
        if age == '(0-2)' or age == '(4-6)':
            how_old = 'child'
            childNum += 1
        else:
            how_old = 'not a child'
            
        face_region = findRegionForFace(faceCenters[i])
        childInFront =  face_region < 2 and how_old == 'child'
        passengerExtra = faceNum > car_passenger_limit
        
        cv2.putText(resultImg, f'{gender}, {how_old}', (faceBox[0], faceBox[1]-10), font, 0.55, (0,255,0), 1, line)
        print()
    
    if childInFront:
        cv2.putText(resultImg, 'Children not allowed in the front seats', (250, 50), font, 0.6, (0, 0, 255), 1, line)
        
    if passengerExtra:
        cv2.putText(resultImg, 'Passenger Limit Exceeded', (250, 25), font, 0.6, (0, 0, 255), 1, line)
    
    cv2.rectangle(resultImg, (x1, y1), (x2, y2), (255, 255, 255), 2, 3)
    cv2.rectangle(resultImg, (x3, y3), (x4, y4), (255, 255, 255), 2, 3)
    
    cv2.putText(resultImg, f'No. of people: {faceNum}', (10, 25), font, 0.6, (0,255,0), 1, line)
    cv2.putText(resultImg, f'No. of children: {childNum}', (10, 50), font, 0.6, (0,255,0), 1, line)    
    cv2.imshow("Video", resultImg)
    # out.write(resultImg)
    
cv2.destroyAllWindows()
# out.release()
