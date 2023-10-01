import React, { useEffect, useRef, useState } from "react";
import styled from "styled-components";
import { keyframes } from "styled-components";
import MyButton from "components/common/StartButton";
import { useNavigate } from "react-router-dom";
import AWS from "aws-sdk";

const fadeIn = keyframes`
  0% {
    opacity: 0;
  }
  50% {
    opacity: 1;
  }
  100% {
    opacity: 1;
  }
`;

const Style = {
  Wrapper: styled.div`
    display: flex;
    justify-content: center;
    flex-direction: column;

    width: 100%;
    height: 100%;

    opacity: 0;
    animation-name: ${fadeIn};
    animation-duration: 3s;
    animation-fill-mode: forwards;
  `,
  TextWrapper: styled.div`
    font-family: NotoSansKR-400;
    font-size: 30px;
    color: black;
    margin: 0 auto;
    margin-bottom: 30px;
  `,
  SmallTextWrapper: styled.div`
    margin-top: 20px;
    font-size: 15px;
    color: #999999;
    display: flex;
    justify-content: center;
    align-items: center;
  `,

  InputWrapper: styled.div`
    display: flex;
    justify-content: center; /* 가로 방향 가운데 정렬 */
    align-items: center; /* 세로 방향 가운데 정렬 */
    margin-top: 10px; /* 위쪽 여백 조정 */
  `,

  InputContainer: styled.div`
    font-family: NotoSansKR-400;
    margin: 10px;
    width: 200px;
    height: 300px;
    font-size: 20px;
    border: 1px solid #999999;
    text-align: center;
    border-radius: 10px;
    padding: 2px;
  `,
  InputText: styled.div`
    font-family: NotoSansKR-500;
    border-bottom: 1px solid #999999;
    padding-bottom: 5px;
  `,
  FileInput: styled.div`
    cursor: pointer;
    width: 100%;
    height: 265px;
    //background-color: #b0b0b0;
    color: #999999;
    display: flex;
    justify-content: center;
    align-items: center;
    //opacity: 0.5;
  `,
  Input: styled.input.attrs({ type: "file", accept: ".jpg, .jpeg, .png" })`
    display: none;
  `,
  TextInput: styled.input.attrs({
    type: "text",
  })`
    margin-left: 30px;
    width: 200px;
    height: 50px;
    font-size: 20px;
    text-align: center;
    font-family: NotoSansKR-500;
  `,

  PreviewImage: styled.img`
    width: 100%;
    height: 100%;
    object-fit: contain;
  `,
  ButtonArea: styled.div`
    margin-top: 60px;
    display: flex;
    justify-content: center;
    align-items: center;
  `,

  CheckInput: styled.input.attrs({ type: "checkbox" })`
    width: 15px;
    height: 15px;
    accent-color: #ff6781;
  `,
};

AWS.config.update({
  region: "보호처리", // 버킷이 존재하는 리전을 문자열로 입력합니다. (Ex. "ap-northeast-2")
  credentials: new AWS.CognitoIdentityCredentials({
    IdentityPoolId: "보호처리", // cognito 인증 풀에서 받아온 키를 문자열로 입력합니다. (Ex. "ap-northeast-2...")
  }),
});

function SelectContent() {
  const [skin, setSkin] = useState(); // 미리보기
  const [skinFile, setSkinFile] = useState(""); // 실제로 aws에 전송되는 것
  const [rb, setRb] = useState();
  const [rbFile, setRbFile] = useState("");
  const [pic, setPic] = useState();
  const [picFile, setPicFile] = useState("");

  const [name, setName] = useState("");
  const [markvu, setMarkvu] = useState(true);

  const fileInput1 = useRef(null);
  const fileInput2 = useRef(null);
  const fileInput3 = useRef(null);

  const navigate = useNavigate();

  function onClick1() {
    fileInput1.current.click();
  }
  function onClick2() {
    fileInput2.current.click();
  }
  function onClick3() {
    fileInput3.current.click();
  }

  const encodeFileToBase64 = (fileBlob, num) => {
    const reader = new FileReader();
    reader.readAsDataURL(fileBlob);
    return new Promise((resolve) => {
      reader.onload = () => {
        if (num == 1) {
          setSkin(reader.result);
        } else if (num == 2) {
          setRb(reader.result);
        } else {
          setPic(reader.result);
        }
        resolve();
      };
    });
  };

  function handleFile1(e) {
    if (e.target.files.length !== 0) {
      encodeFileToBase64(e.target.files[0], 1);
      setSkinFile(e.target.files[0]);
    }
  }

  function handleFile2(e) {
    if (e.target.files.length !== 0) {
      encodeFileToBase64(e.target.files[0], 2);
      setRbFile(e.target.files[0]);
    }
  }

  function handleFile3(e) {
    if (e.target.files.length !== 0) {
      encodeFileToBase64(e.target.files[0], 3);
      setPicFile(e.target.files[0]);
    }
  }

  function onHandleName(e) {
    setName(e.target.value);
    sessionStorage.setItem("name", e.target.value);
  }

  function onHandleMarkVu(e) {
    console.log(markvu);
    setMarkvu(!markvu);
  }

  function checkFiles() {
    console.log(markvu);
    if (markvu === true) {
      // markvu 결과지가 있다면
      if (skin && rb && pic && name) return true;
    } else {
      if (pic && name) return true;
    }

    return false;
  }

  function onClickButton() {
    console.log(name);
    console.log(markvu);
    if (markvu === true) {
      sessionStorage.setItem("isMarkVu", "True");
    } else {
      sessionStorage.setItem("isMarkVu", "False");
    }

    const s3 = new AWS.S3();

    if (checkFiles()) {
      // 증명사진 업로드
      if (picFile) {
        const picParams = {
          Bucket: "보호처리",
          Key: name + "_pic.jpg",
          Body: picFile,
        };
        s3.upload(picParams, function (err, data) {
          if (err) {
            console.log("증명사진 업로드 오류:", err);
          } else {
            console.log("증명사진 업로드 성공:", data);
          }
        });
      }

      // 피부톤 결과지 업로드
      if (skinFile) {
        const skinParams = {
          Bucket: "보호처리",
          Key: name + "_skin.jpg",
          Body: skinFile,
        };
        s3.upload(skinParams, function (err, data) {
          if (err) {
            console.log("피부톤 결과지 업로드 오류:", err);
          } else {
            console.log("피부톤 결과지 업로드 성공:", data);
          }
        });
      }

      // R&B 색소 결과지 업로드
      if (rbFile) {
        const rbParams = {
          Bucket: "보호처리",
          Key: name + "_rb.jpg",
          Body: rbFile,
        };
        s3.upload(rbParams, function (err, data) {
          if (err) {
            console.log("R&B 색소 결과지 업로드 오류:", err);
          } else {
            console.log("R&B 색소 결과지 업로드 성공:", data);
          }
        });
      }

      navigate("/eyebrow");
    } else {
      alert("입력칸을 확인해주세요");
    }
  }

  return (
    <Style.Wrapper>
      <Style.TextWrapper>
        MarkVu 결과지 2장과, 증명사진을 첨부해주세요
        <Style.SmallTextWrapper>
          * MarkVu 결과지는 피부톤, R&B 색소 결과지를 첨부해주세요 <br />*
          증명사진은 눈썹이 잘 보이는 사진으로 첨부해주세요
        </Style.SmallTextWrapper>
      </Style.TextWrapper>

      <Style.TextWrapper>
        귀하의 이름을 적어주세요
        <Style.TextInput onChange={onHandleName} />
      </Style.TextWrapper>

      <Style.InputWrapper>
        <Style.InputContainer>
          <Style.InputText>피부톤 결과지</Style.InputText>
          <Style.FileInput onClick={onClick1}>
            {skin ? (
              <Style.PreviewImage src={skin} alt="preview-skin" />
            ) : (
              <p>여기를 클릭하세요</p>
            )}
          </Style.FileInput>
          <Style.Input ref={fileInput1} onChange={handleFile1}></Style.Input>
        </Style.InputContainer>
        <Style.InputContainer>
          <Style.InputText>R&B 색소 결과지</Style.InputText>
          <Style.FileInput onClick={onClick2}>
            {rb ? (
              <Style.PreviewImage src={rb} alt="preview-rb" />
            ) : (
              <p>여기를 클릭하세요</p>
            )}
          </Style.FileInput>
          <Style.Input ref={fileInput2} onChange={handleFile2}></Style.Input>
        </Style.InputContainer>
        <Style.InputContainer>
          <Style.InputText>증명사진</Style.InputText>
          <Style.FileInput onClick={onClick3}>
            {pic ? (
              <Style.PreviewImage src={pic} alt="preview-pic" />
            ) : (
              <p>여기를 클릭하세요</p>
            )}
          </Style.FileInput>
          <Style.Input ref={fileInput3} onChange={handleFile3}></Style.Input>
        </Style.InputContainer>
      </Style.InputWrapper>
      <Style.SmallTextWrapper>
        마크뷰 결과지가 없습니다
        <Style.CheckInput onClick={onHandleMarkVu} />
      </Style.SmallTextWrapper>
      <Style.ButtonArea onClick={onClickButton}>
        <MyButton text="분석하기 ▶" />
      </Style.ButtonArea>
    </Style.Wrapper>
  );
}

export default SelectContent;
