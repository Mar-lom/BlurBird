import cv2 as cv

videoCap = cv.VideoCapture(0)

if not videoCap.isOpened():
    print("The Camera cant open") #check if the camera can open

while videoCap.isOpened():
    ret, frame = videoCap.read()

    key = cv.waitKey(10)
    if key ==27: #esc
        break

    if ret == True:
        #print("Reading Video")
        #Display the Video
        cv.imshow('Output', frame)
        if cv.waitKey(25) & 0xFF == ord('x'):
            break

    else:
        break
videoCap.release()

cv.destroyAllWindows()


