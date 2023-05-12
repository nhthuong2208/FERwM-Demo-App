import React from "react";
import { Image, Button, Space } from "antd";
import { CameraOutlined, UploadOutlined } from "@ant-design/icons";
import negative_img from "../assets/images/anger-1.jpg";
import neutral_img from "../assets/images/disgust-1.png";
import positive_img from "../assets/images/happy-1.png";
import Header from "../components/Header";
import { Link } from "react-router-dom";
import Footer from "../components/Footer";

const imageIntro = {
  margin: "auto",
  width: "fit-content",
  display: "flex",
  columnGap: "10px",
};

const welcomeText = {
  margin: "auto",
  width: "fit-content",
  marginTop: "10px",
  fontSize: "22px",
  fontWeight: "400",
  letterSpacing: "0",
  color: "#383e45",
  maxWidth: "800px",
  textAlign: "center"
};

const titleText = {
  textAlign: "center",
  padding: "20px 0",
  fontSize: "42px",
  fontWeight: "600",
  letterSpacing: "0",
  color: "#383e45"
}

const Welcome = () => {
  return (
    <>
      <Header />
      <div style={{
        backgroundColor: "#f3f0ec", 
        height: "calc(100vh - 110px)",
        overflow: "none",
        marginTop: "60px"
      }}>
        <h1 style={titleText}>Facial Expresison Recogniton Demo App</h1>
        <div style={imageIntro}>
          <Image width={200} src={negative_img} />
          <Image width={200} src={neutral_img} />
          <Image width={200} src={positive_img} />
        </div>
        <div style={welcomeText}>
          This is a website demo the application of our project: Recognize the
          expression of face that has mask
        </div>
        <Space
          direction="vertical"
          align="center"
          style={{ margin: "auto", marginTop: "10px", width: "100%" }}
        >
          <Link to={"/webcam"}>
            <Button type="primary" icon={<CameraOutlined />}>
              Take image with webcam
            </Button>
          </Link>
          <Link to={"/upload"}>
            <Button type="primary" icon={<UploadOutlined />}>
              Upload Image
            </Button>
          </Link>
        </Space>
      </div>

      <Footer />
    </>
  );
};

export default Welcome;
