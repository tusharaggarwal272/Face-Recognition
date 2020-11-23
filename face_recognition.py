import numpy as np
import cv2
import os
#*************************************
def distance(x1,x2):
    return np.sqrt(((sum(x1-x2))**2).sum())
def knn(train,test,k=5):
    dis=[]
    
    for i in range(train.shape[0]):
        ix=train[i,:-1]
        iy=train[i,-1]
        d=distance(test,ix)
        dis.append([d,iy])
    dk=sorted(dis,key=lambda x:x[0])[:k]
    labels=np.array(dk)[:,-1]
    output=np.unique(labels,return_counts=True)
    index=np.argmax(output[1])
    return output[0][index]

#********
face_data=[]
cap=cv2.VideoCapture(0)
models=cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
skip=0
dataset_path='D://vs code python//Data//'
face_data=[]
labels=[]
class_id=0
names={}
#Data Preparation
for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        names[class_id]=fx[:-4]
        print("loaded"+fx)
        data_item=np.load(dataset_path+fx)
        face_data.append(data_item) 
        target=class_id*np.ones((data_item.shape[0],))
        class_id+=1
        labels.append(target)
face_dataset=np.concatenate(face_data,axis=0)
face_labels=np.concatenate(labels,axis=0).reshape(-1,1)
print(face_dataset.shape)
print(face_labels.shape)
trainset=np.concatenate((face_dataset,face_labels),axis=1)
print(trainset.shape)
#Testing 
while True:
    ret,frame=cap.read()
    if(ret==False):
        continue
    faces=models.detectMultiScale(frame,1.3,5)
    for face in faces:
        x,y,w,h=face
        
        offseet=10
        face_section=frame[y-offseet:y+h+offseet,x-offseet:x+h+offseet]
        face_section=cv2.resize(face_section,(100,100))
        out=knn(trainset,face_section.flatten())
        pred_name=names[int(out)]
        cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2,cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,10),2)
    cv2.imshow("faces",frame)
    Key_press=cv2.waitKey(1)&0xFF
    if(Key_press==ord('q')):
        break
cap.release()
cv2.destroyAllWindows()




    