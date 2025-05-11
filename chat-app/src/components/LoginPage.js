import React, { useContext, useState, useEffect } from 'react';
import './formarea.css'; // Add your custom CSS styles
import { Link, useNavigate } from 'react-router-dom';
import NoteContext from './NoteContext';
import axios from 'axios';

const LoginPage = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const{setLoggedIn,setGlobalUsername} = useContext(NoteContext);
  const navigate = useNavigate();
  const [error, setError] = useState("");
  useEffect(() => {
    if (error) {
      const timer = setTimeout(() => {
        setError("");
      }, 4000); // 4 seconds
  
      return () => clearTimeout(timer); // clean up if the component unmounts early
    }
  }, [error]);  

  const handleLogin = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post("http://localhost:8001/login",{ username: username, password: password , 
      });

      if (response.data.message === "Login successful") {
        localStorage.setItem("username", username); // ✅ Save login
        setLoggedIn(true);
        setGlobalUsername(username);
        navigate("/");
      } else {
        setError(response.data.error || "Login failed. Please try again.");
      }
    } catch (error) {
      console.error("Error during login:", error);
      setError("User not found. Please sign up first.");
    }    
  };

  return (
    <div className="auth-container">
      <section className="auth-section">
        <h1>Login</h1>
        <form onSubmit={handleLogin}>
          <input
            type="text"
            placeholder="username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            required
          />
          <input
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
          <button type="submit" className="new-cta-button">Login</button>
        </form>
        {error && <p className="error-message">{error}</p>}
        <p>
          Don't have an account? <Link to="/signup">Sign Up</Link>
        </p>
      </section>
    </div>
  );
};

export default LoginPage;
