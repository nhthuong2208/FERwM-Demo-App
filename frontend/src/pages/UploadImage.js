import React from "react";
import { useState } from 'react';
import { PlusOutlined, LoadingOutlined, DeleteOutlined, InboxOutlined } from '@ant-design/icons'
import { Button, Col, Row, Upload, Card, message } from 'antd';
import Footer from "../components/Footer";
import Header from "../components/Header";
import baseRequest from "../services/baseRequest";

const containerStyle = {
  padding: "10px",
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  justifyContent: "center", /* Optional: to center horizontally */
  height: "70vh" /* Optional: set the height of the container */
}

const imageStyle = {
  border: "1px dashed #d9d9d9",
  borderRadius: "8px",
  maxWidth: "100%", /* Optional: set the maximum width of the image */
  maxHeight: "100%" /* Optional: set the maximum height of the image */
}

/* const antUploadListItem = {
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  flexDirection: "column",
  width: "102px",
  height: "102px",
  marginInlineEnd: "8px",
  marginBottom: "8px",
  textAlign: "center",
  backgroundColor: "rgba(0, 0, 0, 0.02)",
  border: "1px dashed #d9d9d9",
  borderRadius: "8px",
  cursor: "pointer",
  transition: "border-color 0.3s",
} */

const UploadImage = () => {
  const [imgUrl, setImgUrl] = useState('')
  const [loading, setLoading] = useState(false)
  const [predictData, setPredictData] = useState(null)
  const [resultText, setResultText] = useState('')

  const handleChange = (info) => {
    if (info.file.status === 'uploading') {
      setLoading(true)
      return
    }
    if (info.file.status === 'done') {
      setLoading(false)
      message.success('File uploaded successfully!');
    } else if (info.file.status === 'error') {
      message.error('Failed to upload file!');
      setLoading(false)
    }
  }

  const beforeUpload = (file) => {
    const isJpgOrPng = file.type === 'image/jpeg' || file.type === 'image/png';
    if (!isJpgOrPng) {
      message.error('You can only upload JPG/PNG file!');
    }
    const isLt2M = file.size / 1024 / 1024 < 2;
    if (!isLt2M) {
      message.error('Image must smaller than 2MB!');
    }

    const formData = new FormData()
    formData.append('file', file)

    return new Promise((resolve, reject) => {
      baseRequest.post('upload', formData)
        .then(response => {
          setImgUrl(response.data.url)
          message.success('File uploaded successfully!')
        })
        .catch(error => {
          message.error('Failed to upload file!')
        })
    })
  };

  const handlePredict = () => {
    baseRequest.post('predict', { 'image': imgUrl })
      .then(response => {
        setPredictData(response.data)
        var positive_pred = parseFloat(response.data.positive).toFixed(2)
        var negative_pred = parseFloat(response.data.negative).toFixed(2)
        var neutral_pred = parseFloat(response.data.neutral).toFixed(2)
        let max_val = Math.max(response.data.positive, response.data.negative, response.data.neutral).toFixed(2)
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

  /* const uploadButton = (
    <div style={antUploadListItem}>
      {loading ? <LoadingOutlined /> : <PlusOutlined />}
      <div style={{ marginTop: 8 }}>Upload</div>
    </div>
  ); */

  return (
    <>
      <Header />
      <Row>
        <Col span={12} style={containerStyle}>
          {imgUrl ?
            (<img src={imgUrl} alt="uploaded" style={imageStyle} />) :
            (<Upload.Dragger
              name="file"
              listType="picture"
              disabled={imgUrl ? true : false}
              showUploadList={false}
              action={process.env.REACT_APP_BASE_API_URL + 'upload'}
              beforeUpload={beforeUpload}
              handleChange={handleChange}
            >
              <p className="ant-upload-drag-icon">
                <InboxOutlined />
              </p>
              <p className="ant-upload-text">Click or drag file to this area to upload</p>
              <p className="ant-upload-hint">
                Support for a single upload.
              </p>
            </Upload.Dragger>)}
          <Button
            danger
            type="primary"
            icon={<DeleteOutlined />}
            style={{
              display: imgUrl ? "block" : "none",
              marginTop: "10px"
            }}
            onClick={() => {
              setImgUrl('')
              setPredictData(null)
            }}
          >Remove to pick another image
          </Button>
        </Col>
        <Col span={12}>
          {imgUrl ?
            <Button type="primary" onClick={handlePredict}>
              Predict
            </Button> :
            <div>Drag or Click to choose image to predict</div>
          }
          {predictData ?
            <div style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              height: "60vh"
            }}>
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
                <h1 style={{color: "red", textAlign: "center"}}>{resultText}</h1>
              </Card>
            </div>
            :
            <div></div>
          }
        </Col>
      </Row>
      <Footer />
    </>
  );
}

export default UploadImage;