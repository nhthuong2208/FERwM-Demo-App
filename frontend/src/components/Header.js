import React from "react";
import hcmut_logo from "../assets/images/hcmut.png"
import "../styles/Header.css"

const Header = () => {
    return (
        <div className="logo-header-section">
            <img className="app-logo" src={ hcmut_logo } alt="app-logo" />
            <h1 className="logo">Facial Expression Recognition</h1>
        </div>
    );
}

export default Header;