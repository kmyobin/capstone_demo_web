import Loading from "components/common/Loading";
import React, { useEffect, useState } from "react";
import styled, { css } from "styled-components";
import { keyframes } from "styled-components";
import AWS from "aws-sdk";
import axios from "axios";
import { useNavigate } from "react-router-dom";

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

  ImageContainer: styled.div`
    flex-direction: column;
    margin: 0 auto;
  `,

  ResultsWrapper: styled.div`
    display: flex;
    justify-content: center; /* 가로 방향 가운데 정렬 */
    align-items: center; /* 세로 방향 가운데 정렬 */
    flex-wrap: wrap;
  `,

  ResultWrapper: styled.div`
    width: 180px;
    border: 1px solid #999999;
    padding: 2px;
    margin: 5px;
    ${(props) =>
      props.highlight &&
      css`
        border: 3px solid #ff6781;
      `}
    ${(props) =>
      props.original === true &&
      css`
        border: 3px solid black;
      `}
  `,
  ImageWrapper: styled.img`
    width: 100%;
  `,

  IndexText: styled.div`
    font-family: NotoSansKR-500;
    border-bottom: 1px solid #999999;
    padding-bottom: 5px;
    text-align: center;
    ${(props) =>
      props.original === true &&
      css`
        font-weight: 1000;
      `}
  `,
  TextWrapper: styled.div`
    font-size: 20px;
    font-family: NotoSansKR-500;
    text-align: center;
    margin-bottom: 30px;
  `,
};

AWS.config.update({
  region: "보호처리", // 버킷이 존재하는 리전을 문자열로 입력합니다. (Ex. "ap-northeast-2")
  credentials: new AWS.CognitoIdentityCredentials({
    IdentityPoolId: "보호처리", // cognito 인증 풀에서 받아온 키를 문자열로 입력합니다. (Ex. "ap-northeast-2...")
  }),
});

function EyebrowContent() {
  const name = sessionStorage.getItem("name");
  const isMarkVu = sessionStorage.getItem("isMarkVu");
  const [loading, setLoading] = useState(true);
  const [faceshape, setFaceshape] = useState("");
  const [rcmdIndex, setRcmdIndex] = useState([]);

  const navigate = useNavigate();

  function eyebrow_process() {
    return axios
      .get("/eyebrows", { params: { name: name, isMarkVu: isMarkVu } })
      .then((res) => {
        console.log(res);
        get_faceshape();
      })
      .catch((err) => {
        console.log(err);
        //alert("얼굴이 인식되지 않았습니다. 다른 사진으로 시도해주세요");
        //navigate("/");
      });
  }

  function get_faceshape() {
    axios
      .get("/faceshape", { params: { name: name } })
      .then((res) => {
        console.log(res);
        var fsp = res.data.faceshape;
        if (fsp === "oval") {
          setFaceshape("타원형");
          setRcmdIndex([...rcmdIndex, 6]);
        } else if (fsp === "round") {
          setFaceshape("둥근형");
          setRcmdIndex([...rcmdIndex, 11]);
        } else if (fsp === "square") {
          setFaceshape("각진형");
          setRcmdIndex([...rcmdIndex, 10]);
        } else if (fsp === "heart") {
          setFaceshape("하트형");
          setRcmdIndex([...rcmdIndex, 8, 9]);
        } else {
          setFaceshape("긴형");
          setRcmdIndex([...rcmdIndex, 3, 4]);
        }
      })
      .catch((err) => {
        console.log(err);
        alert("얼굴이 인식되지 않았습니다. 다른 사진으로 시도해주세요");
        navigate("/");
      });
  }

  useEffect(() => {
    eyebrow_process()
      .then((res) => {
        //console.log(res);
        setLoading(false);
      })
      .catch((err) => {
        console.log(err);
      });
  }, []);
  return (
    <Style.Wrapper>
      {!loading ? (
        <>
          <Style.TextWrapper>
            <span style={{ fontWeight: "1000" }}>{name}</span>님은&nbsp;
            <span
              style={{
                color: "#ff6781",
                fontFamily: "NotoSansKR-700",
                fontSize: "25px",
              }}
            >
              {faceshape}
            </span>
            입니다
          </Style.TextWrapper>
          <Style.TextWrapper>아래는 눈썹 예상 결과입니다</Style.TextWrapper>
          <Style.ResultsWrapper>
            <Style.ResultWrapper original={true}>
              <Style.IndexText original={true}>원본</Style.IndexText>
              <Style.ImageWrapper
                src={`https://capstonekmyobinbucket.s3.ap-northeast-1.amazonaws.com/${name}_pic.jpg`}
                alt="preview"
              />
            </Style.ResultWrapper>
            {[...Array(11)].map((_, index) => (
              <Style.ResultWrapper highlight={rcmdIndex.includes(index + 1)}>
                <Style.IndexText>{index + 1}</Style.IndexText>
                <Style.ImageWrapper
                  key={index}
                  src={`https://capstonekmyobinbucket.s3.ap-northeast-1.amazonaws.com/${name}_result_${
                    index + 1
                  }.jpg`}
                  alt="preview"
                />
              </Style.ResultWrapper>
            ))}
          </Style.ResultsWrapper>
        </>
      ) : (
        <Loading />
      )}
    </Style.Wrapper>
  );
}

export default EyebrowContent;
