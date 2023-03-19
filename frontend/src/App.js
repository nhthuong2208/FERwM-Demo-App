import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import UploadImage from './pages/UploadImage';
import WebcamImage from './pages/WebcamImage';
import Welcome from './pages/Welcome';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route exact path="/" element={<Welcome />}/>
        <Route exact path="/webcam" element={<WebcamImage />}/>
        <Route exact path="/upload" element={<UploadImage />}/>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
