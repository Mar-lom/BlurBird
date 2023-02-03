import cv2 as cv
import mediapipe as mp
import numpy as np


mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_drawStyle = mp.solutions.drawing_styles

#debug draw bool
drawit = True
def main():
    videoCap = cv.VideoCapture(0)
    hands = mp_hands.Hands(
        model_complexity = 0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )#setting up params for the hand tracker

    if not videoCap.isOpened():
        print("The Camera cant open") #check if the camera can open

    while True:
        ret, frame = videoCap.read()
        key = cv.waitKey(10)
        if key ==27: #ESC to close window
            break

        frame.flags.writeable = False #makes it so that the image data is passed by reference before inital processing,
                                        #improves performance

        frame =cv.cvtColor(frame, cv.COLOR_BGR2RGB)#convert image to RGB for mediapipe processing
        frame = cv.flip(frame, 1)  # invert the display, so it looks correct left to right
        tracking = hands.process(frame)#tracking sends the converted image data through the mediapipe .Hands() function
                                        #to get landmark data

        frame.flags.writeable = True
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)#converts image back to standard output after landmark data obtained

        if tracking.multi_hand_landmarks: # if the frame returns landmark data->
            for hand_landmarks in tracking.multi_hand_landmarks: #draw each landmark and each connnection
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    #mp_hands.HAND_CONNECTIONS,
                    #mp_drawStyle.get_default_hand_landmarks_style(),
                    #mp_drawStyle.get_default_hand_connections_style()

                )
                bbox = calc_bounding_box(frame,hand_landmarks)

                frame = draw_bbox(drawit,frame,bbox)

        cv.imshow('Output', frame)# output window


    videoCap.release()
    cv.destroyAllWindows()


def calc_bounding_box(window, landmarks):

    window_width, window_height = window.shape[1],window.shape[0] #get the window's width and height

    lm_array = np.empty((0,2), int)

    for _,landmark in enumerate(landmarks.landmark):
        lm_x = min(int(landmark.x * window_width), window_width-1)

        lm_y = min(int(landmark.y *window_height), window_height-1)

        lm_point = [np.array((lm_x,lm_y))]

        lm_array = np.append(lm_array,lm_point, axis=0)

    x,y,w,h = cv.boundingRect(lm_array)


    return[x,y,x+w,y+h]

#draws the bounding box
def draw_bbox(drawit, window,bbox):
    if drawit:

        cv.rectangle(window,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,0),-1)

    return window


if __name__== "__main__":
    main()
