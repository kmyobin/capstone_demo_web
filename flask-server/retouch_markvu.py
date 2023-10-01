import cv2
import numpy as np
import os
from PIL import Image
from PIL import ImageGrab
from pytesseract import *
import sys
import boto3
from botocore.exceptions import NoCredentialsError

# AWS S3 접근 정보 설정
ACCESS_KEY = '보호처리'
SECRET_KEY = '보호처리'
BUCKET_NAME = '보호처리'

# image 다운로드
def image_download(name, markvu):
  OBJECT_NAME = ''
  if markvu=='skin':
      OBJECT_NAME = name+'_skin.jpg' # 원하는 opject 
  elif markvu=='rb' :
      OBJECT_NAME = name+"_rb.jpg"
  else:
      OBJECT_NAME = name+"_pic.jpg"

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

      #cv2.imshow('what', img)
      #cv2.waitKey(0)
      #cv2.destroyAllWindows()

      return img

  except NoCredentialsError:
      print('AWS 자격 증명 정보가 올바르지 않습니다.')

def image_trim(img, x, y, w, h):
   img_trim=img[y:y+h, x:x+w]
   return img_trim

def save_in_bucket(savepath, key):
  # S3 클라이언트 초기화
  s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)
  
  s3.upload_file(savepath, BUCKET_NAME, key)

def resize_image(image):
  original_height, original_width, _ = image.shape

  new_width = 470

  # 새로운 높이 계산
  new_height = int((new_width / original_width) * original_height)

  # 이미지 크기 조정
  resized_image = cv2.resize(image, (new_width, new_height))

  return resized_image
  
if __name__=='__main__':
  name = sys.argv[1]
  isMarkVu=str(sys.argv[2])

  
  if isMarkVu=='True': # 마크뷰 결과지 있으면
    img=image_download(name, 'skin')
    img_rb=image_download(name, 'rb')

    img_trim=image_trim(img, 103, 191, 499, 749) # 얼굴 영상만 추출
    num_trim=image_trim(img, 859, 456, 50, 30) # 숫자 영상만 추출
    # R&B 색소 평균 추출
    num_trim_r=image_trim(img_rb, 860, 454,  55, 30)
    num_trim_b=image_trim(img_rb, 856, 863,  55, 30)

    scale_percent = 200 # 확대할 비율
    width = int(num_trim_r.shape[1] * scale_percent / 100)
    height = int(num_trim_r.shape[0] * scale_percent / 100)  

    width2 = int(num_trim_b.shape[1] * scale_percent / 100)
    height2 = int(num_trim_b.shape[0] * scale_percent / 100)

    dim=(width, height) 
    dim2=(width2, height2)
    num_trim_r=cv2.resize(num_trim_r, dim, interpolation=cv2.INTER_AREA)
    num_trim_b=cv2.resize(num_trim_b, dim2, interpolation=cv2.INTER_AREA)

    custom_config=r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
    text=pytesseract.image_to_string(num_trim, config="kor")
    text_r = pytesseract.image_to_string(num_trim_r, config=custom_config)
                                      #config='--psm 11 -c tessedit_char_whitelist=0123456789')
    text_b = pytesseract.image_to_string(num_trim_b, config=custom_config)
                                      #config='--psm 11 -c tessedit_char_whitelist=0123456789')

    '''
    camera matrix와 distortion coefficients 값을 적절하게 조정
    camera calibration을 통해 camera_matrix, distortion 구함
    '''
    camera_matrix = np.array([[575.440, 0, 182.335],
                              [0, 575.621, 300.01],
                              [0, 0, 1]])
    dist_coeffs = np.array([-0.223114, 0.127922, -0.000141, 0.002677])

    # 왜곡 보정하기
    result_img = cv2.undistort(img_trim, camera_matrix, dist_coeffs)

    txt_name="markvu_result/"+name+"_skin_rb.txt" # 텍스트가 저장될 경로
    with open(txt_name, 'w') as f:
      f.write(text + text_r +text_b) # 피부톤, Red 색소, Brown 색소
    save_in_bucket(txt_name, name+"_skin_rb.txt")

  else: # 마크뷰 결과지 없으면 증사로 얼굴형 분류
    img=image_download(name, 'pic')
    result_img = img

  result_img = resize_image(result_img)
  
  # 가공한 정보 저장하기
  save_result_img="markvu_result/markvu.jpg" # 이미지가 저장될 경로
  cv2.imwrite(save_result_img, result_img)

  save_in_bucket(save_result_img, name+"_markvu.jpg")
  