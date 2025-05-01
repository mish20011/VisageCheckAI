import React, { useContext, useState } from 'react';
import './formarea.css'; // Add your custom CSS styles
import { Link, useNavigate } from 'react-router-dom';
import NoteContext from './NoteContext';
import axios from 'axios';
const LoginPage = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const{setLoggedIn,setGlobalUsername} = useContext(NoteContext);
  const navigate = useNavigate();

  const handleLogin = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post("http://localhost:8001/login",{ username: username, password: password , 
      });

      if (response.data.message === "Login successful.") {
        console.log("Login successful:", response.data);
        setLoggedIn(true);
        const NewUsername = username ; 
        setGlobalUsername(NewUsername);
        navigate("/");
      } else {
        alert(response.data.error);
      }
    } catch (error) {
      console.error("Error during login:", error);
      alert("An error occurred while logging in.");
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
        <p>
          Don't have an account? <Link to="/signup">Sign Up</Link>
        </p>
      </section>
    </div>
  );
};

export default LoginPage;
