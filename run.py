# -*- coding: utf-8 -*-

import cv2
import numpy as np
import math


#Loading Mask RCNN
net = cv2.dnn.readNetFromTensorflow("dnn/frozen_inference_graph_coco.pb",
                                  "dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")


cap = cv2.VideoCapture("los_angeles.mp4")


fbgb = cv2.createBackgroundSubtractorMOG2() # Substraccion de fondo (elementos movimento blanco y fondo en negro)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)) # Kernel se usa para mejorar calidad imagen binaria
color_rect = (0,0,255)



# Initialize count
count = 0
center_points_prev_frame = []

tracking_objects = {}
track_id = 0

frame_count = 0






while True:
    ret, frame = cap.read()
    count += 1
    if not ret:
        break


    x = int(frame.shape[1]*0.9)
    y = int(frame.shape[0]*0.9)

    # Resize de la imagen a 1080*720
    frame_res = cv2.resize(frame, (x,y)) #Cambiamos la resolucion para reducir tamaño video



    # Dibujarmos el contorno de la zona a analizar (Barcelona)
    area_pts_aero = np.array([
        [380,412],
        [700,412], 
        [940,frame_res.shape[0]],
        [140,frame_res.shape[0]]
        ]) 

    # Dibujarmos el contorno de la zona a analizar (los_angeles)
    area_pts_la = np.array([
        [345,300],
        [535,300], 
        [999,frame_res.shape[0]],
        [84,frame_res.shape[0]]
        ]) 

    # Contorno real de la imagen analizada (Barcelona)
    area_pts_real_aero = np.array([
        [350,350],
        [750,350], 
        [1000,frame_res.shape[0]],
        [100,frame_res.shape[0]]
        ]) 

    # Contorno real de la imagen analizada (los_angeles)
    area_pts_real_la = np.array([
        [350,350],
        [750,350], 
        [1000,frame_res.shape[0]],
        [100,frame_res.shape[0]]
        ]) 





    area_pts = area_pts_la
    area_pts_real = area_pts_real_la


    cv2.drawContours(frame_res, [area_pts],-1, color_rect, 2) #Dibujamos rectangulo





    ## Generamos imagen auxiliar para eliminar fondo
    #imAux = np.zeros(shape=(frame_res.shape[:2]), dtype=np.uint8)
    #imAux = cv2.drawContours(imAux, [area_pts_real], -1, (255),-1)
    #imagen_area = cv2.bitwise_and(frame_res, frame_res, mask=imAux)


    imagen_area = frame_res

    height, width, _ = imagen_area.shape




    # Detect objects
    blob = cv2.dnn.blobFromImage(imagen_area, swapRB=True)
    net.setInput(blob)

    boxes, masks = net.forward(["detection_out_final", "detection_masks"])
    detection_count = boxes.shape[2]



    for i in range(detection_count):
        box = boxes[0,0,i]


        #Clases: personas=0, coches=2
        if box[1] == 2:
            x = int(box[3]*width)
            y = int(box[4]*height)
            x2 = int(box[5]*width)
            y2 = int(box[6]*height)

            pnt_x = int(x+((x2-x)/2))
            pnt_y = y2


            
            cv2.rectangle(imagen_area, (x,y), (x2,y2), (255,0,0), 2)
            cv2.circle(imagen_area, (pnt_x,pnt_y) , 5, (0, 0, 255),-1)
        else:
            pass

    cv2.imshow('imagen_area', imagen_area)


    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()