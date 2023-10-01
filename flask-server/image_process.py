import boto3
import numpy as np
from botocore.exceptions import NoCredentialsError
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import PIL.Image
import pyamg
import sys
import cv2
import scipy.sparse
import io

# 눈썹 랜드마크 인덱스
right_eyebrow_landmarks = [300, 293, 334, 296, 336, 285, 276, 283, 282, 295] # 383, 417, 265, 353
left_eyebrow_landmarks = [70, 63, 105, 66, 107, 55, 46, 53, 52, 65] # 156, 193, 35, 124
# 주석 친 랜드마크는 너무 길어서 필요없어서 삭제 처리

# 얼굴형 랜드마크 인덱스
silhouette_landmarks = [
  10,  338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
  397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
  172, 58,  132, 93,  234, 127, 162, 21,  54,  103, 67,  109
]

# AWS S3 접근 정보 설정
ACCESS_KEY = '보호처리'
SECRET_KEY = '보호처리'
BUCKET_NAME = '보호처리'
 

# 1. image 다운로드
def image_download(name):
  OBJECT_NAME = name+'_pic.jpg' # 원하는 opject 
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

      return img

  except NoCredentialsError:
      print('AWS 자격 증명 정보가 올바르지 않습니다.')

# 2. 가져온 img에서 눈썹 위치 추출
def get_eyebrow(image):
  left_coor=[] # 랜드마크 좌표
  right_coor=[] # 랜드마크 좌표
  oval_coor=[] # 얼굴형

  # 랜드마크들의 좌표 알아내기
  detect_landmarks(image, left_landmarks=left_coor, right_landmarks=right_coor, oval_landmarks=oval_coor)

  # 눈썹 영역 좌표 알아내기
  left_eyebrow_xmin, left_eyebrow_ymin, left_eyebrow_xmax, left_eyebrow_ymax = extract_eyebrow_coor(left_coor, padding=4)
  right_eyebrow_xmin, right_eyebrow_ymin, right_eyebrow_xmax, right_eyebrow_ymax = extract_eyebrow_coor(right_coor, padding=4)

  left_eyebrow_coor=[left_eyebrow_xmin, left_eyebrow_ymin, left_eyebrow_xmax, left_eyebrow_ymax]
  right_eyebrow_coor=[right_eyebrow_xmin, right_eyebrow_ymin, right_eyebrow_xmax, right_eyebrow_ymax]

  return left_eyebrow_coor, right_eyebrow_coor

def extract_eyebrow_coor(coors, padding):
  # 눈썹 좌표 초기화
  eyebrow_xmin = float('inf')
  eyebrow_ymin = float('inf')
  eyebrow_xmax = float('-inf')
  eyebrow_ymax = float('-inf')

  # 눈썹 좌표의 최소, 최대 값을 계산
  for coor in coors:
    x, y = coor['x'], coor['y']
    eyebrow_xmin = min(eyebrow_xmin, x)
    eyebrow_ymin = min(eyebrow_ymin, y)
    eyebrow_xmax = max(eyebrow_xmax, x)
    eyebrow_ymax = max(eyebrow_ymax, y)
    
  eyebrow_xmin-=padding
  eyebrow_ymin-=padding
  eyebrow_xmax+=padding
  eyebrow_ymax+=padding

  return eyebrow_xmin, eyebrow_ymin, eyebrow_xmax, eyebrow_ymax

def detect_landmarks(face, left_landmarks, right_landmarks, oval_landmarks):
  mp_face_landmark = mp.solutions.face_mesh
  # 랜드마크 추출
  with mp_face_landmark.FaceMesh(min_detection_confidence=0.1) as face_landmark:
    results = face_landmark.process(face)
 
    # 눈썹 랜드마크들의 좌표 구하기
    get_landmarks_coor(results, face, left_eyebrow_landmarks, left_landmarks)
    get_landmarks_coor(results, face, right_eyebrow_landmarks, right_landmarks)

    # 얼굴형 랜드마크들의 좌표 구하기
    get_landmarks_coor(results, face, silhouette_landmarks, oval_landmarks)

def get_landmarks_coor(face, image, landmarks, coors):
  # landmarks 추출
  if face.multi_face_landmarks:
    for face_landmarks in face.multi_face_landmarks:
      for index in landmarks:
        landmark = face_landmarks.landmark[index]
        x = int(landmark.x * image.shape[1])
        y = int(landmark.y * image.shape[0])                      
        coors.append({'x':x, 'y':y})
        #cv2.circle(image, (x, y), 2, (0, 255, 0), -1)


# 3. 추출한 눈썹으로 blending
def blending_eyebrow(image, name, left_eyebrow_coor, right_eyebrow_coor):
   # 눈썹 영역 크기
   l_width=left_eyebrow_coor[2]-left_eyebrow_coor[0]
   l_height=left_eyebrow_coor[3]-left_eyebrow_coor[1]
   r_width=right_eyebrow_coor[2]-right_eyebrow_coor[0]
   r_height=right_eyebrow_coor[3]-right_eyebrow_coor[1]

   for i in range(1, 12):
        img_target=image # 복사
        # 경로 설정
        m1="final_eyebrows/"+str(i)+"_l_m.jpg"
        s1="final_eyebrows/"+str(i)+"_l.png"
        m2="final_eyebrows/"+str(i)+"_r_m.jpg"
        s2="final_eyebrows/"+str(i)+"_r.png"

        # 읽어들이기
        img_mask1=cv2.imread(m1, cv2.IMREAD_COLOR)
        img_source1=cv2.imread(s1, cv2.IMREAD_COLOR)
        img_mask2=cv2.imread(m2, cv2.IMREAD_COLOR)
        img_source2=cv2.imread(s2, cv2.IMREAD_COLOR)

        # target 눈썹 영역 bounding box 크기만큼 resize
        img_mask1=cv2.resize(img_mask1, (l_width, l_height)) # 왼쪽 
        img_source1=cv2.resize(img_source1, (l_width,l_height))
        img_mask2=cv2.resize(img_mask2, (r_width, r_height)) # 오른쪽
        img_source2=cv2.resize(img_source2, (r_width,r_height))

        # 색 영역 BGR -> RGB
        img_mask1 = cv2.cvtColor(img_mask1, cv2.COLOR_BGR2RGB)
        img_source1 = cv2.cvtColor(img_source1, cv2.COLOR_BGR2RGB)
        img_mask2 = cv2.cvtColor(img_mask2, cv2.COLOR_BGR2RGB)
        img_source2 = cv2.cvtColor(img_source2, cv2.COLOR_BGR2RGB)
        img_target = cv2.cvtColor(img_target, cv2.COLOR_BGR2RGB)

        # target 눈썹 시작 위치를 offset으로 설정 (행, 열)
        img_ret = blend(img_target, img_source1, img_mask1, offset=(left_eyebrow_coor[1],left_eyebrow_coor[0])) # y, x 171 126
        img_ret = blend(img_ret, img_source2, img_mask2, offset=(right_eyebrow_coor[1],right_eyebrow_coor[0])) # y, x 172 228


        img_ret = PIL.Image.fromarray(np.uint8(img_ret))

        svpth="result/"+name+"_result_"+str(i)+".jpg"
        img_ret.save(svpth)

        save_in_bucket(svpth, name, i)

# pre-process the mask array so that uint64 types from opencv.imread can be adapted
def prepare_mask(mask):
    if type(mask[0][0]) is np.ndarray:
        result = np.ndarray((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if sum(mask[i][j]) > 0:
                    result[i][j] = 1
                else:
                    result[i][j] = 0
        mask = result
    return mask

def blend(img_target, img_source, img_mask, offset=(0, 0)):
    # compute regions to be blended
    region_source = (
            max(-offset[0], 0),
            max(-offset[1], 0),
            min(img_target.shape[0]-offset[0], img_source.shape[0]),
            min(img_target.shape[1]-offset[1], img_source.shape[1]))
    region_target = (
            max(offset[0], 0),
            max(offset[1], 0),
            min(img_target.shape[0], img_source.shape[0]+offset[0]),
            min(img_target.shape[1], img_source.shape[1]+offset[1]))
    region_size = (region_source[2]-region_source[0], region_source[3]-region_source[1])

    print(region_size)

    # clip and normalize mask image
    img_mask = img_mask[region_source[0]:region_source[2], region_source[1]:region_source[3]]
    img_mask = prepare_mask(img_mask)
    img_mask[img_mask==0] = False
    img_mask[img_mask!=False] = True

    
    # create coefficient matrix
    A = scipy.sparse.identity(np.prod(region_size), format='lil')
    for y in range(region_size[0]):
        for x in range(region_size[1]):
            #print(x,y)
            if img_mask[y,x]:
                index = x+y*region_size[1]
                A[index, index] = 4
                if index+1 < np.prod(region_size):
                    A[index, index+1] = -1
                if index-1 >= 0:
                    A[index, index-1] = -1
                if index+region_size[1] < np.prod(region_size):
                    A[index, index+region_size[1]] = -1
                if index-region_size[1] >= 0:
                    A[index, index-region_size[1]] = -1
    A = A.tocsr()
    
    # create poisson matrix for b
    P = pyamg.gallery.poisson(img_mask.shape)

    # for each layer (ex. RGB)
    for num_layer in range(img_target.shape[2]):
        # get subimages
        t = img_target[region_target[0]:region_target[2],region_target[1]:region_target[3],num_layer]
        s = img_source[region_source[0]:region_source[2], region_source[1]:region_source[3],num_layer]
        t = t.flatten()
        s = s.flatten()

        # create b
        b = P * s
        for y in range(region_size[0]):
            for x in range(region_size[1]):
                if not img_mask[y,x]:
                    index = x+y*region_size[1]
                    b[index] = t[index]

        # solve Ax = b
        x = pyamg.solve(A,b,verb=False,tol=1e-10)

        # assign x to target image
        x = np.reshape(x, region_size)
        x[x>255] = 255
        x[x<0] = 0
        x = np.array(x, img_target.dtype)
        img_target[region_target[0]:region_target[2],region_target[1]:region_target[3],num_layer] = x

    return img_target

def save_in_bucket(savepath, name, num):
  # S3 클라이언트 초기화
  s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)
  
  key=name+'_result_'+str(num)+".jpg"
  s3.upload_file(savepath, BUCKET_NAME, key)

if __name__ == '__main__':
  '''
  1. 그냥 일단 다 하기
  '''
  name = sys.argv[1]
  image = image_download(name)
     
  l, r = get_eyebrow(image)
  blending_eyebrow(image, name, l, r)

