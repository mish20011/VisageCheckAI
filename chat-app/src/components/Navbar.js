import React, { useContext, useState} from 'react';
import { Menu } from 'lucide-react';
import './Navbar.css';
import { Link, useNavigate } from 'react-router-dom';
import NoteContext from './NoteContext.js'; 
import myLogo from './images/icon_6110f3bc411f9.png'
function Navbar() {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const navigate = useNavigate() ; 
  const {globalUsername , setGlobalUsername } = useContext(NoteContext) ;
  const toggleMenu = () => {
    setIsMenuOpen(!isMenuOpen);
  };
  const getInitials = (name) => {
    return name ? name[0].toUpperCase() : '';
  };
  const handleLogout = () => {
    setGlobalUsername(null); // Clear the global username
    navigate('/'); // Redirect to login page
  };
  const scrollToSection = (id) => {
    const section = document.getElementById(id);
    if (section) {
      section.scrollIntoView({ behavior: 'smooth' });
      setIsMenuOpen(false); // Close the menu if it's open
    }
  };
  return (
    <nav className="navbar">
      <div className="nav-container">
        <Link to="/" className="nav-logo"
         onClick={() => {
          window.scrollTo({ top: 0, behavior: 'smooth' })}}>
        <img src = {myLogo} className='my-logo'/>
          <span>Derma Doc AI</span>
        </Link>

        <button className="nav-mobile-toggle" onClick={toggleMenu}>
          <Menu size={24} />
        </button>

        <div className={`nav-links ${isMenuOpen ? 'active' : ''}`}>
        <>
        <p onClick={() => 
        {navigate('/') ; 
        scrollToSection('about')}}
          className='logout-button'>About</p>
  <p onClick={() => {
    navigate('/') ; 
    scrollToSection('services')}}
    className='logout-button'>Services</p>
  <p onClick={() => {
    navigate('/') ; 
    scrollToSection('contact')}}
    className='logout-button'>Contact</p> </>
          {globalUsername ? (
            // If globalUsername is not null, display username with the icon and logout button
            <>
            <div className="user-info">
              <div className="user-icon">
                {getInitials(globalUsername)}
              </div>
              <span className="username">{globalUsername}</span>
            </div>
            <div onClick={handleLogout}
            className='logout-button'>
            Logout
            </div></>
          ) : (
            // If globalUsername is null, show the Login and Sign Up links
            <>
              <Link to="/login">Login</Link>
              <Link to="/signup">Sign Up</Link>
            </>
          )}
        </div>
      </div>
    </nav>
  );
}

export default Navbar;