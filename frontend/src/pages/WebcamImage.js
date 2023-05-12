import React, { useState, useRef, useEffect } from "react";
import Webcam from "react-webcam";
import baseRequest from "../services/baseRequest";
import { Checkbox, Col, Row } from "antd";
import Header from "../components/Header";
import Footer from "../components/Footer";

const videoConstraints = {
  width: 400,
  height: 400,
  facingMode: "user",
};

const imageStyle = {
  border: "1px dashed #d9d9d9",
  borderRadius: "8px",
  width: "390px",
  marginTop: "20px",
};

const containerStyle = {
  padding: "10px",
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  justifyContent: "center" /* Optional: to center horizontally */,
  height: "50vh" /* Optional: set the height of the container */,
};

const WebcamImage = () => {
  const webcamRef = useRef(null);
  const [predictData, setPredictData] = useState(null);
  const [showGrad, setShowGrad] = useState(false);
  const [showTwo, setShowTwo] = useState(false);

  const runCoco = () => {
    // Loop and detect hands
    showGrad ? setShowTwo(true) : setShowTwo(false);
    setInterval(() => {
      handleCallAPI();
    }, 3000);
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

      baseRequest
        .post("predict-video", { image: img })
        .then((response) => {
          setPredictData(response.data);
        })
        .catch((error) => {
          console.log(error);
        });
    }
  };

  useEffect(() => {
    runCoco();
  }, [showGrad]);

  return (
    <>
      <Header />
      <div
        style={{
          backgroundColor: "#f3f0ec",
          height: "calc(100vh - 110px)",
          overflow: "none",
          marginTop: "60px",
        }}
      >
        {showTwo ? (
          <Row style={{ marginTop: "10px" }}>
            <Col
              span={12}
              style={{
                display: "flex",
                flexDirection: "column",
                justifyContent: "center",
                alignItems: "center",
              }}
            >
              <Webcam
                audio={false}
                ref={webcamRef}
                videoConstraints={videoConstraints}
                screenshotFormat="image/jpeg"
                mirrored={true}
              />
              <div style={{ marginTop: "15px" }}>
                <Checkbox
                  checked={showGrad}
                  disabled={predictData ? false : true}
                  onChange={(e) => setShowGrad(e.target.checked)}
                >
                  Show Grad-CAM to verify
                </Checkbox>
              </div>
            </Col>
            <Col span={12} style={containerStyle}>
              <img
                src={`data:image/jpeg;base64,${predictData.gradcam}`}
                style={imageStyle}
                alt="uploaded"
              />
            </Col>
          </Row>
        ) : (
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              justifyContent: "center",
              alignItems: "center",
              marginTop: "10px",
            }}
          >
            <Webcam
              audio={false}
              ref={webcamRef}
              videoConstraints={videoConstraints}
              screenshotFormat="image/jpeg"
              mirrored={true}
            />
            <div style={{ marginTop: "15px" }}>
              <Checkbox
                checked={showGrad}
                disabled={predictData ? false : true}
                onChange={(e) => setShowGrad(e.target.checked)}
              >
                Show Grad-CAM to verify
              </Checkbox>
            </div>
          </div>
        )}
        {predictData && (
          <div style={{ textAlign: "center", marginTop: "40px" }}>
            <div>
              Negative:{" "}
              <span style={{ fontWeight: "bold" }}>
                {parseFloat(predictData.probs.negative).toFixed(4)}
              </span>
              , Positive:{" "}
              <span style={{ fontWeight: "bold" }}>
                {parseFloat(predictData.probs.positive).toFixed(4)}
              </span>
              , Neutral:{" "}
              <span style={{ fontWeight: "bold" }}>
                {parseFloat(predictData.probs.neutral).toFixed(4)}
              </span>{" "}
            </div>
            <div
              style={{
                fontWeight: "bold",
                fontSize: "30px",
                color:
                  predictData.predict === "negative"
                    ? "red"
                    : predictData.predict === "positive"
                    ? "green"
                    : "#8B8000",
              }}
            >
              {predictData.predict}
            </div>
          </div>
        )}
      </div>
      <Footer />
    </>
  );
};

export default WebcamImage;
