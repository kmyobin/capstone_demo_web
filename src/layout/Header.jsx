import React from "react";
import styled from "styled-components";
import "../App.css";
import { useNavigate } from "react-router-dom";
import { UndoOutlined } from "@ant-design/icons";

const Style = {
  Header: styled.div`
    position: sticky;
    top: 0;
    height: 80px;
    width: 100vw;
    background: #ff6781;
    font-family: NotoSansKR-700;
    font-size: 24px; //1.5rem;
    //padding-left: 20px;
    color: #ffffff;
    display: flex;
    align-items: center;
    justify-content: space-between;
  `,
  Logo: styled.div`
    cursor: pointer;
    @media only screen and (min-width: 769px) {
      // 노트북, 데스크탑
      float: left;
      padding: 1%; // Logo에 padding 추가하면 가로 스크롤바 해결
      width: 50%;
    }
  `,
  Redo: styled.div`
    padding-right: 3%;
    padding-left: 3%;
    -webkit-transform: rotate(90deg);
    transform: rotate(90deg);
    font-size: 100%;
  `,
};

function Header() {
  const navigate = useNavigate();

  function onClickUndo() {
    navigate("/");
  }

  function onClickLogo() {
    navigate("/");
  }

  let Lo = window.location.href;
  return (
    <Style.Header>
      <Style.Logo onClick={onClickLogo}>정보통신종합설계</Style.Logo>
      <Style.Redo>
        {/* 선택 페이지나 결과 페이지에서만 Redo 버튼 활성화 */}
        {(Lo.includes("/select") ||
          Lo.includes("/result") ||
          Lo.includes("/eyebrow")) && <UndoOutlined onClick={onClickUndo} />}
      </Style.Redo>
    </Style.Header>
  );
}

export default Header;
