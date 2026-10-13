import os

import cv2

# 1. Set RTSP_URL in the environment with your camera's username, password, and the specific stream path,
# e.g. rtsp://<user>:<password>@<host>:554/<stream_path>
# Common stream paths include: /live/ch0, /h264Preview_01_main, /live0, or /stream1
RTSP_URL = os.environ.get("RTSP_URL", "")


def main():
    if not RTSP_URL:
        print("Error: Set the RTSP_URL environment variable to your camera's stream URL.")
        return

    # Initialize the video capture stream
    cap = cv2.VideoCapture(RTSP_URL)
    
    if not cap.isOpened():
        print("Error: Could not open the RTSP video stream. Verify your URL and network path.")
        return

    print("Streaming started successfully. Press 'q' on the video window to quit.")
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to grab frame. Stream might have disconnected.")
            break
            
        # Display the live feed in a desktop window
        cv2.imshow('IP Camera RTSP Stream', frame)
        
        # Break the loop immediately if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Clean up and release system assets
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
