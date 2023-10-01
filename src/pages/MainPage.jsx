import PageTemplate from "layout/PageTemplate";
import React from "react";
import MainContent from "components/main/MainContent";

function MainPage() {
  return <PageTemplate content={<MainContent />} />;
}

export default MainPage;
