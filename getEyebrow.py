from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

'''
눈썹, 얼굴 외곽 추출 (468 landmarks)
'''
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

maximum_width=0
maximum_height=0

def draw_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style())
    
    solutions.drawing_utils.draw_landmarks( 
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
    
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_iris_connections_style())

  return annotated_image

def plot_face_blendshapes_bar_graph(face_blendshapes):
  # Extract the face blendshapes category names and scores.
  face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
  face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
  # The blendshapes are ordered in decreasing score value.
  face_blendshapes_ranks = range(len(face_blendshapes_names))

  fig, ax = plt.subplots(figsize=(12, 12))
  bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
  ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
  ax.invert_yaxis()

  # Label each bar with values
  for score, patch in zip(face_blendshapes_scores, bar.patches):
    plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

  ax.set_xlabel('Score')
  ax.set_title("Face Blendshapes")
  plt.tight_layout()
  plt.show()
  
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

'''
def assemble_eyebrow(original_face, left_eyebrow, right_eyebrow):
  # 원본 이미지와 눈썹 이미지 로드
  original_face=cv2.imread('img.jpg')
  left_eyebrow = cv2.imread("left_eyebrow.jpg")
  right_eyebrow = cv2.imread("right_eyebrow.jpg")

  # 눈썹 이미지의 크기와 위치를 확인
  left_eyebrow_height, left_eyebrow_width, _ = left_eyebrow.shape
  right_eyebrow_height, right_eyebrow_width, _ = right_eyebrow.shape

  # 눈썹의 크기 비율 계산
  left_eyebrow_ratio = left_eyebrow_width / left_eyebrow_height
  right_eyebrow_ratio = right_eyebrow_width / right_eyebrow_height

  # 눈썹이 합성될 위치 설정
  left_eyebrow_points = np.array([[x1, y1], [x2, y2], [x3, y3]], dtype=np.float32)
  right_eyebrow_points = np.array([[x1, y1], [x2, y2], [x3, y3]], dtype=np.float32)

  # 합성될 얼굴 이미지의 눈썹 부분에 affine 변환 적용
  left_eyebrow_transformed = cv2.warpAffine(left_eyebrow, cv2.getAffineTransform(left_eyebrow_points, new_points), (new_width, new_height))
  right_eyebrow_transformed = cv2.warpAffine(right_eyebrow, cv2.getAffineTransform(right_eyebrow_points, new_points), (new_width, new_height))

  # 합성될 얼굴 이미지에 눈썹 이미지 합성
  result = original_face.copy()
  result[y1:y2, x1:x2] = left_eyebrow_transformed
  result[y1:y2, x3:x4] = right_eyebrow_transformed

  # 결과 이미지 출력
  cv2.imshow("Result", result)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
'''

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
    
    '''
    # 랜드마크들의 좌표 출력
    print('왼쪽 눈썹 랜드마크의 좌표')
    print(left_landmarks)
    print('오른쪽 눈썹 랜드마크의 좌표')    
    print(right_landmarks)    
    print('얼굴형 랜드마크의 좌표')
    print(oval_landmarks)
    '''
    
def draw_boundingbox(image, bbox):
  cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)   

def extract_eyebrow(image, bbox):
  eyebrow=image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
  return eyebrow

def draw_mask(width, height, maskpath):
  black_image = np.zeros((height, width, 3), dtype=np.uint8)
  cv2.imwrite(maskpath, black_image)

def demo(realpath):
  face=cv2.imread(realpath)
  #mask=cv2.imread(maskpath)
  
  left_coor=[] # 랜드마크 좌표
  right_coor=[] # 랜드마크 좌표
  oval_coor=[] # 얼굴형

  # 랜드마크들의 좌표 알아내기
  detect_landmarks(face, left_landmarks=left_coor, right_landmarks=right_coor, oval_landmarks=oval_coor)

  # 눈썹 영역 좌표 알아내기
  left_eyebrow_xmin, left_eyebrow_ymin, left_eyebrow_xmax, left_eyebrow_ymax = extract_eyebrow_coor(left_coor, padding=4)
  right_eyebrow_xmin, right_eyebrow_ymin, right_eyebrow_xmax, right_eyebrow_ymax = extract_eyebrow_coor(right_coor, padding=4)

  left_eyebrow_coor=[left_eyebrow_xmin, left_eyebrow_ymin, left_eyebrow_xmax, left_eyebrow_ymax]
  right_eyebrow_coor=[right_eyebrow_xmin, right_eyebrow_ymin, right_eyebrow_xmax, right_eyebrow_ymax]
  
  '''
  # 눈썹 영역의 최대 너비, 높이 알아내기
  global maximum_width
  global maximum_height
  maximum_width = max(right_eyebrow_coor[2]-right_eyebrow_coor[0], maximum_width)
  maximum_width = max(maximum_width, left_eyebrow_coor[2]-left_eyebrow_coor[0])
  maximum_height = max(right_eyebrow_coor[3]-right_eyebrow_coor[1], maximum_height)
  maximum_height = max(maximum_height, left_eyebrow_coor[3]-left_eyebrow_coor[1])
  
  
  # 눈썹 bounding box 그리기
  #draw_boundingbox(face, left_eyebrow_coor)
  #draw_boundingbox(face, right_eyebrow_coor)
    
  # 눈썹 bounding box 추출하기
  #left_eyebrow_real=extract_eyebrow(face, left_eyebrow_coor)
  #right_eyebrow_real=extract_eyebrow(face, right_eyebrow_coor)

  #draw_mask(left_eyebrow_real.shape[1], left_eyebrow_real.shape[0], "l_mask.png")
  #draw_mask(right_eyebrow_real.shape[1], right_eyebrow_real.shape[0], "r_mask.png")
  '''

  return left_eyebrow_coor, right_eyebrow_coor


if __name__ == '__main__':
  realpath='img6.jpg'
  #maskpath='face_parsing/face-parsing.PyTorch/res/test_res/'+str(i)+'.png'
  l,r = demo(realpath=realpath)

  l_width=l[2]-l[0]
  l_height=l[3]-l[1]

  r_width=r[2]-r[0]
  r_height=r[3]-r[1]
  print(l)
  print(r)

  print(l_width, l_height)
  print(r_width, r_height)
  # 필요한 것 : source의 눈썹 위치와 크기

  
  
    