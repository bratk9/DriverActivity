import cv2
import os
import pickle
import face_recognition

def isDriver(x, y):
    return (x < x2) and (y > y1)

# rectangles of driver seat and passenger seat respectively
reg_point = [ [0, 185, 250, 480], [390, 185, 640, 480] ]
x1 = reg_point[0][0]
y1 = reg_point[0][1]
x2 = reg_point[0][2]
y2 = reg_point[0][3]
x3 = reg_point[1][0]
y3 = reg_point[1][1]
x4 = reg_point[1][2]
y4 = reg_point[1][3]


if not os.path.exists('known_faces'):
    os.makedirs('known_faces')
KNOWN_FACES_DIR = 'known_faces'
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
video = cv2.VideoCapture(0)
id = input("Enter ID for face: ")
if not os.path.exists(f'{KNOWN_FACES_DIR}/{id}'):
    os.makedirs(f'{KNOWN_FACES_DIR}/{id}')
    count = 0
else:
    count = len([i for i in os.listdir(f'{KNOWN_FACES_DIR}/{id}')])

if os.path.exists('faces'):
    infile = open('faces','rb')
    known_faces = pickle.load(infile)
    infile.close()
else:
    known_faces = []
if os.path.exists('names'):
    infile = open('names','rb')
    known_names = pickle.load(infile)
    infile.close()
else:
    known_names = []
print('Make sure the Face Capture window is active by clicking on it.')
print('Press space to capture a frame.')
print('Take atleast 5-10 pictures from different angles for best results.')
while True:
    ret, image = video.read()
    save = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)        #Haar_Cascades_Implementation
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    locations = []
    for (x,y,w,h) in faces:
        driver = isDriver( (x + w//2), (y + h//2) )  # rectangle center point is passed
        if driver:
            cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
            # print(image.shape)
            locations.append((y,x+w,y+h,x))
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2, 3)
    cv2.rectangle(image, (x3, y3), (x4, y4), (255, 255, 255), 2, 3)
    cv2.imshow("Face Capture", image)
    k =  cv2.waitKey(1) & 0xFF
    if k == 32:         #SPACE
        if(len(faces)>0):
            encoding = face_recognition.face_encodings(image,locations)[0]
            known_faces.append(encoding)
            known_names.append(id)
            cv2.imwrite(f'{KNOWN_FACES_DIR}/{id}/{count}.jpg',save);
            count+=1
            print(f"{count} frame(s) captured.")
        else:
            print('Driver face not detected. Please try again.')
    elif k == 27:       #ESC
        break
cv2.destroyAllWindows()
outfile = open('faces','wb')   
pickle.dump(known_faces,outfile)
outfile.close()
outfile = open('names','wb')   
pickle.dump(known_names,outfile)
outfile.close()