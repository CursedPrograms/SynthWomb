import cv2

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture video from webcam
    ret, frame = cap.read()

    # Display the video feed
    cv2.imshow('Webcam Feed', frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
