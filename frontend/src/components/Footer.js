import React from "react";

const footerStyle = {
    position: "absolute",
    bottom: "0",
    height: "50px",
    width: "100%",
    backgroundColor: "gray",
    lineHeight: "50px",
    textAlign: "right",
    display: "flex", 
    justifyContent:"space-between"
}

const Footer = () => {
    return (
        <div style={footerStyle}>
            <div style={{marginLeft: "10px"}}>Explore our Project: <a 
                    href="https://github.com/nh0znoisung/FER"
                    target="_blank"
                    rel="noopener noreferrer"
                    style={{textDecoration:"none"}}>Github</a>
            </div>
            <div style={{marginRight: "10px"}}>FERwM ©2023 Created by Thuong and Tuan</div>
        </div>
    );
}

export default Footer;