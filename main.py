import cv2 as cv
import mediapipe as mp


mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_drawStyle = mp.solutions.drawing_styles


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
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawStyle.get_default_hand_landmarks_style(),
                    mp_drawStyle.get_default_hand_connections_style()

                )
        cv.imshow('Output', frame)# output window


    videoCap.release()

    cv.destroyAllWindows()


if __name__== "__main__":
    main()
