import React from "react";
import { Link, NavLink } from "react-router-dom";
import hcmut_logo from "../assets/images/hcmut.png";
import "../styles/Header.css";

const Header = () => {
  return (
    <header style={{
      boxShadow: "0 3px 6px 0 rgb(50 50 50 / 30%)",
      boxSizing: "border-box",
      height: "60px",
      zIndex: "1041",
      position: "fixed",
      top: "0",
      left: "0",
      right: "0"
    }}>
      <div className="logo-header-section">
        <Link to="/">
          <img className="app-logo" src={hcmut_logo} alt="app-logo" />
        </Link>
        <nav style={{
          flexGrow: "3",
        }}>
          <ul className="nav-style">
            <NavLink to="/upload" className={({isActive}) => (isActive ? "active" : null)} style={{textDecoration: "none" }}>
              <li>UPLOAD IMAGE</li>
            </NavLink>
            <NavLink to="/webcam" className={({isActive}) => (isActive ? "active" : null)} style={{textDecoration: "none"}}>
              <li>USING WEBCAM</li>
            </NavLink>
          </ul>
        </nav>
      </div>
    </header>
  );
};

export default Header;
