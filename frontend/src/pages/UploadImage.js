import React from "react";
import { useState } from "react";
import { Button, Col, Row, Checkbox, Spin } from "antd";
import { DeleteOutlined } from "@ant-design/icons";
import Footer from "../components/Footer";
import Header from "../components/Header";
import baseRequest from "../services/baseRequest";
import UploadFile from "../components/UploadFile";

const checkBoxStyle = {
  display: "flex",
  flexDirection: "column",
  justifyContent: "center",
  alignItems: "center",
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

const UploadImage = () => {
  const [imgUrl, setImgUrl] = useState("");
  const [predictData, setPredictData] = useState(null);
  const [showGrad, setShowGrad] = useState(false);
  const [showTwo, setShowTwo] = useState(false);
  const [loading, setLoading] = useState(false);

  // Change probs
  const handlePredict = () => {
    setLoading(true);
    setImgUrl(localStorage.getItem("url"));

    showGrad ? setShowTwo(true) : setShowTwo(false);
    baseRequest
      .post("predict", { image: localStorage.getItem("url") })
      .then((response) => {
        setLoading(false);
        setPredictData(response.data);
      })
      .catch((error) => {
        console.log(error);
      });
  };

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
        {loading && (
          <Spin
            tip="Loading"
            size="large"
            style={{
              marginTop: "100px",
            }}
          >
            <div className="content" />
          </Spin>
        )}
        {showTwo && imgUrl && !loading ? (
          <>
            <Row>
              <Col
                span={12}
                style={{
                  display: "flex",
                  flexDirection: "column",
                  justifyContent: "center",
                  alignItems: "center",
                }}
              >
                <UploadFile />
                <Button
                  danger
                  type="primary"
                  icon={<DeleteOutlined />}
                  style={{
                    display: imgUrl ? "block" : "none",
                  }}
                  onClick={() => {
                    setImgUrl("");
                    localStorage.setItem("url", "");
                    setPredictData(null);
                  }}
                >
                  Remove to pick another image
                </Button>
              </Col>
              <Col span={12} style={containerStyle}>
                <img
                  src={`data:image/jpeg;base64,${predictData.gradcam}`}
                  style={imageStyle}
                  alt="uploaded"
                />
              </Col>
            </Row>
            <div style={{ textAlign: "center", marginTop: "20px" }}>
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
          </>
        ) : (
          <>
            <UploadFile />
            <div style={checkBoxStyle}>
              <Checkbox checked={showGrad} onChange={(e) => setShowGrad(e.target.checked)}>
                Show Grad-CAM to verify
              </Checkbox>
              <Button
                style={{ marginTop: "10px" }}
                type="primary"
                onClick={handlePredict}
              >
                Predict
              </Button>
              <Button
                danger
                type="primary"
                icon={<DeleteOutlined />}
                style={{
                  marginTop: "20px",
                  display: imgUrl ? "block" : "none",
                }}
                onClick={() => {
                  setImgUrl("");
                  localStorage.setItem("url", "");
                  setPredictData(null);
                }}
              >
                Remove to pick another image
              </Button>
            </div>

            {predictData && (
              <div style={{ textAlign: "center", marginTop: "20px" }}>
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
          </>
        )}
      </div>

      <Footer />
    </>
  );
};

export default UploadImage;
