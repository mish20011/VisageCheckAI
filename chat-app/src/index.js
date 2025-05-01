import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import reportWebVitals from './reportWebVitals';
import NoteState from './components/NoteState';
import {
  BrowserRouter as Router,
  Route,
  Routes,
} from 'react-router-dom';
import HomePage from './components/HomePage';
import Navbar from './components/Navbar';
import LoginPage from './components/LoginPage';
import SignupPage from './components/SignUpPage';
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <NoteState>
      <Router>
         <Navbar/>
        <Routes>
          <Route path="/App" element={<App/>} />
          <Route path="/" element={<HomePage/>} />
          <Route path="/Login" element={<LoginPage/>} />
          <Route path="/SignUp" element = {<SignupPage/>}/>
        </Routes>
      </Router>
    </NoteState>
  </React.StrictMode>
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
