import React, { useState, useRef, useEffect } from "react";
import Webcam from "react-webcam"
import baseRequest from "../services/baseRequest";
import { Card } from "antd";
import Header from "../components/Header";
import Footer from "../components/Footer";

const videoConstraints = {
  width: 400,
  height: 400,
  facingMode: 'user',
}

const WebcamImage = () => {
  const webcamRef = useRef(null);
  const [predictData, setPredictData] = useState(null)
  const [resultText, setResultText] = useState('')

  const runCoco = () => {
    //  Loop and detect hands
    setInterval(() => {
      handleCallAPI()
    }, 1000);
  };

  // Change probs
  const handleCallAPI = () => {
    if (
      typeof webcamRef.current !== "undefined" &&
      webcamRef.current !== null &&
      webcamRef.current.video.readyState === 4
    ) {
      // Get Video Properties
      const img = webcamRef.current.getScreenshot();

      baseRequest.post('predict-video', { 'image': img })
        .then(response => {
          setPredictData(response.data.probs)
          var positive_pred = parseFloat(response.data.probs.positive).toFixed(2)
          var negative_pred = parseFloat(response.data.probs.negative).toFixed(2)
          var neutral_pred = parseFloat(response.data.probs.neutral).toFixed(2)
          let max_val = Math.max(response.data.probs.positive, response.data.probs.negative, response.data.probs.neutral).toFixed(2)
          switch (max_val) {
            case positive_pred:
              setResultText('positive')
              break
            case negative_pred:
              setResultText('negative')
              break
            case neutral_pred:
              setResultText('neutral')
              break
          }
        })
        .catch(error => {
          console.log(error)
        })
    }
  }

  useEffect(() => {
    runCoco()
  }, [])

  return (
    <>
      <Header />
      <div style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        columnGap: "50px",
        height: "60vh"
      }}>
        <Webcam
          audio={false}
          ref={webcamRef}
          videoConstraints={videoConstraints}
          screenshotFormat="image/jpeg"
          mirrored={true} />
        {predictData !== null ?
          <Card
            title="Result"
            headStyle={{
              textAlign: "center"
            }}
            style={{
              width: 300,
            }}
          >
            <p>Negative: {predictData.negative}</p>
            <p>Positive: {predictData.positive}</p>
            <p>Neutral: {predictData.neutral}</p>
            <h1 style={{ color: "red", textAlign: "center" }}>{resultText}</h1>
          </Card> : <div></div>
        }
      </div>
      <Footer />
    </>
  );
}

export default WebcamImage;