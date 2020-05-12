import os
import pickle
import cv2
import face_recognition
import shutil
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

KNOWN_FACES_DIR = 'known_faces'
while(True):
    print('Enter option: ')
    print('1. Train Model')
    print('2. Test Model')
    print('3. Re-train from directory')
    print('4. Remove directory')
    print('5. Exit')
    i = int(input("Input (1/2/3/4/5) : "))
    if i == 1:
        os.system('python captureFace.py')
    elif i == 2:
        os.system('python recogniseFace.py')
    elif i == 3:
        if os.path.exists('faces'):
            os.remove('faces')
        if os.path.exists('names'):
            os.remove('names')
        print('Trained data removed successfully.')
        known_faces = []
        known_names = []
        for name in os.listdir(KNOWN_FACES_DIR):
            print(f'Checking in {name}.')
            for filename in os.listdir(f'{KNOWN_FACES_DIR}/{name}'):
                image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{name}/{filename}')
                locations = []
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)        #Haar_Cascades_Implementation
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                try:
                    for (x,y,w,h) in faces:
                        locations.append((y,x+w,y+h,x))
                    encoding = face_recognition.face_encodings(image,locations)[0]
                    known_faces.append(encoding)
                    known_names.append(name)
                    print(f'{filename} loaded.')
                except:
                    print(f'{filename} skipped, due to unrecognised face.')
        outfile = open('faces','wb')   
        pickle.dump(known_faces,outfile)
        outfile.close()
        outfile = open('names','wb')   
        pickle.dump(known_names,outfile)
        outfile.close()
    elif i == 4:
        if not os.path.exists(KNOWN_FACES_DIR):
            print('Directory does not exist.')
        else:
            confirm = input("This will remove all your captured face data. Are you sure you wish to continue? (Y/N) : ")
            if confirm == 'Y' or confirm == 'y':
                shutil.rmtree(KNOWN_FACES_DIR)
                print('Deletion successful.')
            else:
                print('Deletion cancelled.')    
    elif i == 5:
        break
    else:
        print('Invalid option.')