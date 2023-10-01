import React from "react";
import { Button } from "antd";
import styled from "styled-components";

const StyledButton = styled(Button)`
  :where(.css-dev-only-do-not-override-k7429z).ant-btn-default:not(
      :disabled
    ):hover {
    color: white;
    border-color: white;
  }
  background-color: #ff6781;
  color: white;
  font-family: NotoSansKR-500;
  width: 351px;
  height: 50px;
  border-radius: 1000px;
  font-size: 18px;
`;

function StartButton({ text }) {
  return <StyledButton>{text}</StyledButton>;
}

export default StartButton;
