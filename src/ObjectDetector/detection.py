from ultralytics import YOLO
import cv2 as cv
from vidgear.gears import CamGear

class YoloV8:
    def __init__(self, modelname="yolov8n.pt", segment=False, pose=False):
        if segment and pose:
            print("Only Segmentation Or Positioning is possible.")
            return
            
        elif segment:
            self.model = YOLO(modelname.split('.')[0]+'-seg.pt')
        elif pose:
            self.model = YOLO(modelname.split('.')[0]+'-pose.pt')
        else:
            self.model = YOLO(modelname)
        
    def predictImage(self, image, conf=0.60):
        result = self.model(image, conf=conf)
        annotatedImage = result[0].plot()
        annotatedImage = cv.resize(annotatedImage, (int(annotatedImage.shape[1]/2), int(annotatedImage.shape[0]/2)))
        cv.imshow('Results', annotatedImage)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def predictVideo(self, videoPath, conf=0.60):
        cap = cv.VideoCapture(videoPath)

        if not cap.isOpened():
            print("Error: Unable to open video file.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = self.model(frame, conf=conf)
            annotatedImage = result[0].plot()
            annotatedImage = cv.resize(annotatedImage, (int(annotatedImage.shape[1]/2), int(annotatedImage.shape[0]/2)))
            cv.imshow('Results', annotatedImage)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv.destroyAllWindows()
        
    
    def predictYouTubeVideo(self, YTURL, skip_frames=1,FRAME_WIDTH = 1280,FRAME_HEIGHT=720,FPS=30,conf=0.60):
        
        options = {"CAP_PROP_FRAME_WIDTH":FRAME_WIDTH, "CAP_PROP_FRAME_HEIGHT":FRAME_HEIGHT, "CAP_PROP_FPS":FPS}
        
        stream = CamGear(source=YTURL, stream_mode=True, **options).start()
    
        frame_count = 0
        while True:
            frame = stream.read()
            if frame is None:
                break
            
            if frame_count % skip_frames == 0:
                result = self.model(frame, conf=conf)
                annotatedImage = result[0].plot()
                annotatedImage = cv.resize(annotatedImage, (int(annotatedImage.shape[1]/1.5), int(annotatedImage.shape[0]/1.5)))
                cv.imshow('Results', annotatedImage)
    
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
                
            frame_count += 1
    
        cv.destroyAllWindows()
        stream.stop()
