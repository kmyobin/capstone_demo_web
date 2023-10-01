import React, { useEffect, useState } from "react";
import styled from "styled-components";
import MyButton from "components/common/StartButton";
import { useNavigate } from "react-router-dom";
import { keyframes } from "styled-components";

const fadeIn1 = keyframes`
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

const fadeIn2 = keyframes`
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
    width: 100%;
    height: 100%;
    display: flex;
    justify-content: center;
    align-items: center;
    flex-direction: column;
  `,
  TextArea: styled.div`
    font-family: NotoSansKR-500;
    font-size: 36px;
    line-height: 50px;
    text-align: center;
    margin: 40px 0px;
    margin-bottom: 150px;
  `,
  ButtonArea: styled.div`
    position: fixed;
    bottom: 95px;
  `,

  Sentence1: styled.div`
    opacity: 0;
    animation-name: ${fadeIn1};
    animation-duration: 3s;
    animation-fill-mode: forwards;
    margin-bottom: 50px;
  `,

  Sentence2: styled.div`
    opacity: 0;
    animation-name: ${fadeIn2};
    animation-duration: 3s;
    animation-delay: 2s;
    animation-fill-mode: forwards;
  `,
};

function MainContent() {
  const navigate = useNavigate();

  return (
    <Style.Wrapper>
      <Style.TextArea>
        <Style.Sentence1>나에게 맞는</Style.Sentence1>
        <Style.Sentence2>눈썹과 헤어스타일이 궁금하다면?</Style.Sentence2>
      </Style.TextArea>
      <Style.ButtonArea onClick={() => navigate("/select")}>
        <MyButton text="스타일링 받아보기 ▶" />
      </Style.ButtonArea>
    </Style.Wrapper>
  );
}

export default MainContent;
