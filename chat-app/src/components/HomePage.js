import React, {useContext, useState} from 'react';
import './HomePage.css';
import HeroVideo from '../videos/my_video.mp4';
import NoteContext from './NoteContext';
import { useNavigate } from 'react-router-dom';
import map from './images/generic-3d-map-doctor-hospital-600nw-2077489207.webp';
import aiImg from './images/healthcare-workers-ai-1359494953.webp';
import { Twitter, Facebook, Instagram, Linkedin, Github } from 'lucide-react';

function HomePage() {
  const [result, setResult] = useState("");
  const [showAlert, setShowAlert] = useState(false);
  const{loggedIn} = useContext(NoteContext);
  const Navigate = useNavigate();
  
  const startButton = () => {
    if(loggedIn){
      Navigate('/App') ;  
    }
    else{
      Navigate('/Login')
    }
  }

  const onSubmit = async (event) => {
    event.preventDefault();
    setResult("Sending...");
    setShowAlert(true);

    const formData = new FormData(event.target);
    formData.append("access_key", "4031c5fa-649a-4285-9b8b-9859937c04d5");

    try {
      const response = await fetch("https://api.web3forms.com/submit", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (data.success) {
        setResult("Form Submitted Successfully!");
        event.target.reset();
      } else {
        setResult(data.message || "Failed to submit the form.");
      }
    } catch (error) {
      setResult("An error occurred. Please try again.");
      console.error("Submission error:", error);
    }

    setTimeout(() => setShowAlert(false), 5000);
  };

  return (
    <div className="homepage">
      <section className="hero">
        <video className="hero-video" src={HeroVideo} autoPlay muted loop />
        <div className="hero-content">
          <h1 className="hero-hed">Derma Doc AI</h1>
          <h2>Your ultimate solution for skin health</h2>
          <p className="hero-p">
            Upload images, input symptoms, and receive personalized treatment plans along with recommendations for nearby dermatologists.
          </p>
          <button className="cta-button" onClick={startButton}>Get Started</button>
        </div>
      </section>

      <div className="content-wrapper">
        <section className="about" id="about">
          <h2>About Derma Doc AI</h2>
          <div className="about-content">
            <div className="about-text">
              <h3>Our Mission</h3>
              <p>
                At Derma Doc AI, we're revolutionizing skin health care through the power of artificial intelligence. Our mission is to make professional-grade skin analysis accessible to everyone, anywhere, at any time.
              </p>
              
              <h3>Why Choose Us?</h3>
              <div className="features-grid">
                <div className="feature">
                  <h4>Advanced AI Technology</h4>
                  <p>Our state-of-the-art AI algorithms are trained on millions of dermatological images, ensuring accurate and reliable analysis of skin conditions.</p>
                </div>
                <div className="feature">
                  <h4>Instant Analysis</h4>
                  <p>Get immediate insights about your skin condition without the wait. Our AI provides real-time analysis and recommendations within seconds.</p>
                </div>
                <div className="feature">
                  <h4>Expert Network</h4>
                  <p>Connect with certified dermatologists in your area through our extensive network of healthcare professionals.</p>
                </div>
                <div className="feature">
                  <h4>Privacy First</h4>
                  <p>Your health data is precious. We implement bank-grade security measures to ensure your information remains private and secure.</p>
                </div>
              </div>
            </div>
            <div className="my_images">
              <img src={map} alt="Location Map" className="map-pic"/>
              <img src={aiImg} alt="AI Healthcare" className="ai-pic"/>
            </div>
          </div>
        </section>

        <section className="services" id="services">
  <h2>Our Services</h2>
  <p className="services-intro">
    Discover a wide range of features designed to provide comprehensive skin health solutions. From AI-powered diagnostics to personalized recommendations, we've got you covered.
  </p>
  <div className="services-grid">
    <div className="service1">
      <h4>AI-Powered Image Analysis</h4>
      <p>Utilize advanced AI to accurately identify skin conditions with unparalleled precision.</p>
    </div>
    <div className="service2">
      <h4>Symptom-Based Diagnostic Tools</h4>
      <p>Comprehensive tools to analyze your symptoms and provide a complete health overview.</p>
    </div>
    <div className="service3">
      <h4>Find Nearby Dermatologists</h4>
      <p>Locate certified dermatologists in your area for professional consultations.</p>
    </div>
    <div className="service4">
      <h4>Real-Time AI Predictions</h4>
      <p>Receive instant insights about potential skin issues and preventive care suggestions.</p>
    </div>
    <div className="service5">
      <h4>Personalized Skincare </h4>
      <p>Get tailored product suggestions to suit your unique skin needs.</p>
    </div>
    <div className="service6">
      <h4>24/7 Virtual Assistance</h4>
      <p>Enjoy round-the-clock support for all your skin health inquiries.</p>
    </div>
  </div>
</section>

      </div>


      <footer className="social-footer">
      <section id="contact" className="contact">
        <h2>Contact Us</h2>
        <p>Have questions or feedback? We'd love to hear from you!</p>
        <form className="contact-form" onSubmit={onSubmit}>
          <input type="text" name="name" placeholder="Your Name" required />
          <input type="email" name="email" placeholder="Your Email" required />
          <textarea name="message" placeholder="Your Message" required></textarea>
          <button type="submit">Send Message</button>
        </form>
      </section>
        <div className="social-container">
          <h3>Connect With Us</h3>
          <div className="social-links">
            <a href="https://twitter.com/dermadocai" target="_blank" rel="noopener noreferrer">
              <Twitter size={24} />
            </a>
            <a href="https://facebook.com/dermadocai" target="_blank" rel="noopener noreferrer">
              <Facebook size={24} />
            </a>
            <a href="https://instagram.com/dermadocai" target="_blank" rel="noopener noreferrer">
              <Instagram size={24} />
            </a>
            <a href="https://linkedin.com/company/dermadocai" target="_blank" rel="noopener noreferrer">
              <Linkedin size={24} />
            </a>
            <a href="https://github.com/dermadocai" target="_blank" rel="noopener noreferrer">
              <Github size={24} />
            </a>
          </div>
          <p className="copyright">© 2025 Derma Doc AI. All rights reserved.</p>
        </div>
      </footer>

      {showAlert && (
        <div className="alert-box">
          {result}
        </div>
      )}
    </div>
  );
}

export default HomePage;