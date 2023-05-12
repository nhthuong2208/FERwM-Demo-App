import React, { useState } from 'react'
import { Upload, message } from 'antd'
import { InboxOutlined } from '@ant-design/icons'
import baseRequest from '../services/baseRequest'

message.config({
  top: 100
});

const containerStyle = {
  padding: "10px",
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  justifyContent: "center", /* Optional: to center horizontally */
  height: "50vh" /* Optional: set the height of the container */
}

const imageStyle = {
  border: "1px dashed #d9d9d9",
  borderRadius: "8px",
  maxWidth: "100%", /* Optional: set the maximum width of the image */
  maxHeight: "100%" /* Optional: set the maximum height of the image */
}

const UploadFile = () => {
  const [imgUrl, setImgUrl] = useState('')

  const handleChange = (info) => {
    if (info.file.status === 'uploading') {
      return
    }
    if (info.file.status === 'done') {
      message.success('File uploaded successfully!');
    } else if (info.file.status === 'error') {
      message.error('Failed to upload file!');
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
          localStorage.setItem("url", response.data.url)
          message.success('File uploaded successfully!')
        })
        .catch(error => {
          message.error('Failed to upload file!')
        })
    })
  };
  return (
    <div style={containerStyle}>
      {(localStorage.getItem("url") !== null && localStorage.getItem("url") !== "") ?
          (<img src={localStorage.getItem("url")} alt="uploaded" style={imageStyle} />) :
          (<Upload.Dragger
            name="file"
            listType="picture"
            disabled={(localStorage.getItem("url") !== null && localStorage.getItem("url") !== "") ? true : false}
            showUploadList={false}
            action={process.env.REACT_APP_BASE_API_URL + 'upload'}
            beforeUpload={beforeUpload}
            handleChange={handleChange}
            style={{padding: "10px 50px"}}
          >
            <p className="ant-upload-drag-icon">
              <InboxOutlined />
            </p>
            <p className="ant-upload-text">Click or drag file to this area to upload</p>
            <p className="ant-upload-hint">
              Support for a single upload.
            </p>
          </Upload.Dragger>)}
          
    </div>
  )
}

export default UploadFile