import cv2
import boto3
import numpy as np
from botocore.exceptions import NoCredentialsError

# AWS S3 접근 정보 설정
ACCESS_KEY = '보호처리'
SECRET_KEY = '보호처리'
BUCKET_NAME = '보호처리'
OBJECT_NAME = '보호처리'

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

    # 이미지 표시
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

except NoCredentialsError:
    print('AWS 자격 증명 정보가 올바르지 않습니다.')
