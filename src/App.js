import MainPage from "pages/MainPage";
import "App.css";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import SelectPage from "pages/SelectPage";
import EyebrowPage from "pages/EyebrowPage";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<MainPage />} />
        <Route path="/select" element={<SelectPage />} />
        <Route path="/eyebrow" element={<EyebrowPage />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
