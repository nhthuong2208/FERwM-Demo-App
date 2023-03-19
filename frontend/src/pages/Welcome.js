import React from "react";
import { Image, Button, Space } from "antd"
import { CameraOutlined, UploadOutlined } from "@ant-design/icons"
import negative_img from "../assets/images/Steve_Patterson_0001.jpg"
import Header from "../components/Header";
import { Link } from "react-router-dom";
import Footer from "../components/Footer"

const imageIntro = {
    margin: "auto",
    width: "fit-content",
    display: "flex",
    columnGap: "10px"
}

const welcomeText = {
    margin: "auto",
    width: "fit-content",
    marginTop: "10px"
}

const Welcome = () => {
    return (
        <>
            <Header />
            <div style={imageIntro}>
                <Image
                    width={200}
                    src={negative_img}
                />
                <Image
                    width={200}
                    src={negative_img}
                />
                <Image
                    width={200}
                    src={negative_img}
                />
            </div>
            <div style={welcomeText}>This is a website demo the application of our project: Recognize the expression of face that has mask</div>
            <Space direction="vertical" align="center" style={{margin: "auto", marginTop: "10px", width: "100%"}}>
                <Link to={"/webcam"}>
                    <Button type="primary" icon={<CameraOutlined />}>Take image with webcam</Button>
                </Link>
                <Link to={"/upload"}>
                    <Button type="primary" icon={<UploadOutlined />}>Upload Image</Button>
                </Link>
            </Space>
            <Footer />
        </>
    );
}

export default Welcome;