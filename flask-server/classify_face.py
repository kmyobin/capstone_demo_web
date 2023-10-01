#This script runs the re-trained Inception model to classify a single or a batch of images
import boto3
from botocore.exceptions import NoCredentialsError
import subprocess
from PIL import Image
import matplotlib.pyplot as plt
import pathlib
from datetime import datetime
import time
import tensorflow as tf, sys
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import cv2
import urllib.parse

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

name = sys.argv[1]
#name='김효빈'
current_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.abspath(os.path.join(current_path, '..'))
InputPath= os.path.abspath(os.path.join(current_path, 'result'))
ResultPath= os.path.abspath(os.path.join(current_path, 'classify',name+'_face_shape.txt'))

def classify_image(image_path, model_path, labels_path):
    # Read in the image_data
    time_start = time.monotonic()
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=True))

    encoded_path = urllib.parse.quote(image_path)

    image_data = tf.io.gfile.GFile(encoded_path, 'rb').read()

    # Loads label file, strips off carriage return
    label_lines = ['heart', 'oblong', 'oval', 'round', 'square']
    #label_lines = [line.rstrip() for line 
    #                   in tf.io.gfile.GFile(labels_path)]

    # Unpersists graph from file
    with tf.io.gfile.GFile(model_path, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    with sess.as_default() as sess: 
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        
        predictions = sess.run(softmax_tensor, \
                 {'DecodeJpeg/contents:0': image_data})
        # Sort to show labels of first prediction in order of confidence
        
        
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        #print(top_k, label_lines)
        output_label = ""
        
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            output_label = output_label + human_string + "({0:.4f})".format(score) + " "
            if score == 1:
                with open(ResultPath, "w") as f:
                    f.write(label_lines[node_id])
            
        
        output_label = output_label + " Runtime: " + "{0:.2f}".format(time.monotonic()-time_start) + "s"
    
    sess.close()

# AWS S3 접근 정보 설정
ACCESS_KEY = '보호처리'
SECRET_KEY =  '보호처리'
BUCKET_NAME = '보호처리'

def save_in_bucket(savepath, name):
	# S3 클라이언트 초기화
	s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)
		
	key=name+'_face_shape.txt'
	s3.upload_file(savepath, BUCKET_NAME, key)

# image 다운로드
def image_download(name):
  OBJECT_NAME = name+'_markvu.jpg' # 원하는 opject 
  # S3 클라이언트 초기화
  s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)

  try:
      # S3 버킷에서 이미지 파일 다운로드
      response = s3.get_object(Bucket=BUCKET_NAME, Key=OBJECT_NAME)
      file_content = response['Body'].read()

      # 바이트 데이터를 numpy 배열로 변환
      np_array = np.frombuffer(file_content, np.uint8)

      # OpenCV로 이미지 열기
      img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

      cv2.imwrite('classify/markvu.jpg', img)

  except NoCredentialsError:
      print('AWS 자격 증명 정보가 올바르지 않습니다.')

if __name__ == '__main__':
    name = sys.argv[1]
    #name='김효빈'
    model_dir = os.path.abspath(os.path.join(current_path, 'faceclassification','result'))  
    imagedir = InputPath

    model_path = os.path.abspath(os.path.join(model_dir, 'retrained_graph.pb'))
    labels_path = os.path.abspath(os.path.join(model_dir, 'retrained_labels.txt'))
    batch_run = 0
            
    if (batch_run == 0):
        image_download(name)
        img_path='classify/markvu.jpg'

        print("Processing ", "...", end=' ', sep='') 
        classify_image(img_path, model_path, labels_path)
        save_in_bucket(ResultPath, name) # txt 파일 bucket에 저장