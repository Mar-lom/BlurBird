import cv2

videoCap = cv2.VideoCapture(0)

if not videoCap.isOpened():
    print("The Camera cant open") #check if the camera can open

while videoCap.isOpened():
    ret, frame = videoCap.read()
    if ret == True:
        #print("Reading Video")
        #Display the Video
        cv2.imshow('Output', frame)
        if cv2.waitKey(25) & 0xFF == ord('x'):
            break
    else:
        break
videoCap.release()

cv2.destroyAllWindows()