import os
# import ssl
# print(ssl.OPENSSL_VERSION)
from ultralytics import YOLO
import cv2
import numpy as np

VIDEOS_DIR = os.path.join('.', 'videos')

video_path = os.path.join(VIDEOS_DIR, 'alpaca1.mp4')
video_path_out = '{}_out.mp4'.format(video_path)

cap = cv2.VideoCapture(1)
ret, frame = cap.read()
H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

model_path = os.path.join('.', 'runs', 'detect', 'train10', 'weights', 'best.pt')

# Load a model
#model = YOLO(model_path)  # load a custom model
model = YOLO('yolo11n-seg.pt') # load pre-trained model
threshold = 0.5

while ret:

    results = model(frame)[0]
    # print(results.names)
    # break
    masks = results.masks
    if masks is not None:
        
        mask_array = masks.data.cpu().numpy()  # Chuyển sang numpy
        # Tạo ảnh màu overlay
        overlay = frame.copy()

        for mask in mask_array:
            mask = (mask > 0.5).astype(np.uint8) * 255  # Chuyển thành ảnh nhị phân
            colored_mask = cv2.applyColorMap(mask, cv2.COLORMAP_COOL)  # Tạo màu
            #frame = cv2.addWeighted(overlay, 0.7, colored_mask, 0.3, 0)  # Trộn với ảnh gốc
            colored_mask = cv2.addWeighted(overlay, 0.8, colored_mask, 0.2, 0)
            frame[mask>0] =  colored_mask[mask > 0]
        #cv2.imshow("Segmentation Mask", overlay)    
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.imshow('frame', frame)
    out.write(frame)
    ret, frame = cap.read()
    
    key = cv2.waitKey(1)
    if key & 0xFF == ord('x'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()