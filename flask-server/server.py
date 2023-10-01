from flask import Flask, request
import subprocess
import boto3
from botocore.exceptions import NoCredentialsError

app = Flask(__name__)

@app.route('/eyebrows', methods=['GET'])
def eyebrows():
    name=request.args.get('name')
    isMarkVu=request.args.get('isMarkVu')
    
    # 1. 마크뷰 결과지에 따른 calib 수행 (아니면 그냥 그대로 저장)
    subprocess.run(["ex/Scripts/python", "retouch_markvu.py", name, isMarkVu], capture_output=True, text=True) # 마크뷰 결과지 보정(마크뷰 없으면)
    # 2. 얼굴형으로 분류 실행
    subprocess.run(["ex/Scripts/python", "classify_face.py", name], capture_output=True, text=True) # 얼굴형 분류
    # 3. 눈썹 생성
    subprocess.run(["ex/Scripts/python", "image_process.py", name], capture_output=True, text=True)

    return {"message" : "data received success"}

@app.route('/faceshape', methods=['GET'])
def get_faceshape():
    name=request.args.get('name')

    faceshape=txt_download(name)

    return {"faceshape" : faceshape}

def txt_download(name):
    # AWS S3 접근 정보 설정
    ACCESS_KEY = '보호처리'
    SECRET_KEY = '보호처리'
    BUCKET_NAME = '보호처리'
    OBJECT_NAME = name+'_face_shape.txt' # 원하는 opject 
    # S3 클라이언트 초기화
    s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)

    try:
        # S3 버킷에서 이미지 파일 다운로드
        response = s3.get_object(Bucket=BUCKET_NAME, Key=OBJECT_NAME)
        file_content = response['Body'].read().decode('utf-8')

        # 읽어온 텍스트 파일 내용을 변수에 저장
        text_variable = file_content

        return text_variable

    except NoCredentialsError:
        print('AWS 자격 증명 정보가 올바르지 않습니다.')      


    
           
@app.route('/')
def hello():
    return "HELLO WORLD"

if __name__ == "__main__":
    app.run(debug = True)