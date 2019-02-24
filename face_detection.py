import cv2


#create a classifier for face and eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# reading the image
img = cv2.imread("C:/Users/Tejas/Desktop/0031.jpg")

#converting the image to gray
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# find the co-ordinates of the face
faces = face_cascade.detectMultiScale(gray_img , scaleFactor=1.25 , minNeighbors=5)

for x,y,w,h in faces:
    img = cv2.rectangle(img, (x,y), (x+w-0,y+h-0),(0,255,0),3)

resized = cv2.resize(img,(int(img.shape[1]/2),int(img.shape[0]/2)))

# find the co-ordinates of the eyes
eyes = eye_cascade.detectMultiScale(gray_img, scaleFactor=1.25, minNeighbors= 5)

for x,y,w,h in eyes:
    img = cv2.rectangle(img, (x,y), (x+w,y+h),(255,0,0),3)

resized = cv2.resize(img,(int(img.shape[1]/2),int(img.shape[0]/2)))


cv2.imshow("gray",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
