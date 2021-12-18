import cv2


#Loading Mask RCNN
net = cv2.dnn.readNetFromTensorflow("dnn/frozen_inference_graph_coco.pb",
                                  "dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")


#Load image
img = cv2.imread("road.jpg")
height, width, _ = img.shape

# Detect objects
blob = cv2.dnn.blobFromImage(img, swapRB=True)
net.setInput(blob)

boxes, masks = net.forward(["detection_out_final", "detection_masks"])
detection_count = boxes.shape[2]



for i in range(detection_count):
    box = boxes[0,0,i]


    if box[1] == 0:
        x = int(box[3]*width)
        y = int(box[4]*height)
        x2 = int(box[5]*width)
        y2 = int(box[6]*height)



        cv2.rectangle(img, (x,y), (x2,y2), (255,0,0), 2)
    else:
        pass



cv2.imshow("Image", img)
cv2.waitKey(0)