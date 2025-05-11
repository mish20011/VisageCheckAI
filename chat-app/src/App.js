import React, { useContext, useState, useEffect, useRef } from "react";
import { useCallback } from "react";
import NoteContext from "./components/NoteContext";
import './App.css'; 
import axios from 'axios';
// import {Mic,X} from 'lucide-react';
import googlePoint from './components/images/googlePoint.png';
import imgIcon from './components/images/add_photo_alternate_24dp_666666_FILL0_wght400_GRAD0_opsz24.png';
import CropImageModal from "./components/CropImageModal";
import { Trash } from 'react-feather';
import ConfirmDialog from './components/ConfirmDialog';


function App() {
  const { globalUsername, setGlobalUsername } = useContext(NoteContext);
  const [query, setQuery] = useState('');
  const [file, setFile] = useState(null);
  const [mainRes, setMainRes] = useState([]);
  const [loading, setLoading] = useState(false);
  const [userLocation, setUserLocation] = useState(null);
  const [filePreview, setFilePreview] = useState(null);
  const [showConfirm, setShowConfirm] = useState(false);
  const [selectedImage, setSelectedImage] = useState(null);
  const [showCropper, setShowCropper] = useState(false);
  const chatBottomRef = useRef(null);


  useEffect(() => {
    const savedUsername = localStorage.getItem("username");
    if (savedUsername) {
      setGlobalUsername(savedUsername);
    }
  }, [setGlobalUsername]);
  
    
  useEffect(() => {
    getUserLocation();
  }, []);
  const fetchAllMessages = useCallback(async () => {
    try {
      const response = await axios.post("http://localhost:8001/getMessages", {
        username: globalUsername
      });
      const data = response.data;
      setMainRes(data.myData);
    } catch (error) {
      console.log(error);
    }
  }, [globalUsername]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
    useEffect(() => {
      fetchAllMessages();
    }, [fetchAllMessages]);
    


  useEffect(() => {
    return () => {
      if (filePreview) {
        URL.revokeObjectURL(filePreview);
      }
      mainRes.forEach(item => {
        if (item.imageUrl) {
          URL.revokeObjectURL(item.imageUrl);
        }
      });
    };
  }, [filePreview, mainRes]);
  const saveMessage = async (dataPack) => {
    try {
        if (dataPack.imageFile) {
            const base64Image = await convertFileToBase64(dataPack.imageFile);
            dataPack = {
                ...dataPack,
                imageBase64: base64Image,
            };
        }
        
        dataPack.username = globalUsername;
        const response = await axios.post("http://localhost:8001/saveMessage", dataPack);
        console.log("Message saved to database:", response.data);
    } catch (error) {
        console.error("Error saving message:", error);
        if (error.response) {
            console.error("Server response error:", error.response.data);
        }
    }
};

  useEffect(() => {
    if (chatBottomRef.current) {
      chatBottomRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [mainRes]);


  const getUserLocation = () => {
    if ("geolocation" in navigator) {
      navigator.geolocation.getCurrentPosition(
        async (position) => {
          const lat = position.coords.latitude;
          const long = position.coords.longitude;
          
          try {
            const apiKey = 'e1bbf146ddd64a619a3344e5182b1ce7';
            const url = `https://api.geoapify.com/v1/geocode/reverse?lat=${lat}&lon=${long}&apiKey=${apiKey}`;
            
            const response = await axios.get(url);
            const result = response.data.features[0].properties;
            console.log(`location results are : ${JSON.stringify(result)}`) ; 
            console.log(`location results are : ${result.county}`) ; 
            console.log(`location results are : ${result.state_district}`) ; 
            const city = result.city || 'Bengaluru'; // Default city
            const locality = result.state_district || 'Indiranagar'; // Default locality
            setUserLocation({
              latitude: lat,
              longitude: long,
              city: city,
              locality: locality
            });
          } catch (error) {
            console.error("Error fetching reverse geocoding data:", error);
          }
        },
        (error) => {
          console.error("Error getting location:", error);
        }
      );
    }
  };

  const formatTreatment = (treatmentText) => {
    if (!treatmentText) {
      return <p className="treatment-text">No description available.</p>;
    }
  
    // Split the text into paragraphs by newlines
    const paragraphs = treatmentText.split("\n").filter(paragraph => paragraph.trim() !== "");
  
    return paragraphs.map((para, index) => {
      // Regular expression to match text between '**' for bold formatting
      const parts = para.split(/\*\*(.*?)\*\*/g);
  
      return (
        <p key={index} className="treatment-text">
          {parts.map((part, idx) => {
            // If the part is between '**', wrap it in a <strong> tag to make it bold
            if (idx % 2 === 1) {
              return <strong key={idx}>{part}</strong>;
            }
            // If it's not bold text, return it as normal text
            return <span key={idx}>{part}</span>;
          })}
        </p>
      );
    });
  };
  const convertFileToBase64 = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = (error) => reject(error);
      reader.readAsDataURL(file);
    });
  };
  const renderFunc = () => {
    return (
      <>
        {mainRes.map((item, index) => (
          <React.Fragment key={index}>
            {item.imageBase64 ? (
              <img 
                src={filePreview || item.imageBase64}
                alt="Query" 
                className="query-image"
              />
            ) : (
              <div className="query-box" id="my-query">
                <p className="query">{item.query}</p>
              </div>
            )}
            <div className="res-desc-box">
              {console.log(item.res)}
              {item.res !== "Disease might be" ? <h1 className="res-title">{`RESULT: ${item.res.toUpperCase()}`}</h1> : ""}
              <div className="treatment">{formatTreatment(item.desc)}</div>
            </div>
            {item.doctors && item.doctors.length > 0 && (
              <div className="doctor-details">
                <h2>Recommended Doctors</h2>
                {item.doctors.map((doctor, docIndex) => (
                  <div key={docIndex} className="doctor-card">
                    <h3>{doctor.name}</h3>
                    <p><strong>Qualifications:</strong> {doctor.qualifications || "Not available"}</p>
                    <div className="clinic-details">
                      {Array.isArray(doctor.clinics) ? (
                        doctor.clinics.map((clinic, clinicIndex) => (
                          <div key={clinicIndex} className="clinic">
                            <a
                              href={userLocation 
                                ? `https://www.google.com/maps/dir/${userLocation.latitude},${userLocation.longitude}/${encodeURIComponent(typeof clinic === 'string' ? clinic : clinic.address)}`
                                : `https://www.google.com/maps/search/${encodeURIComponent(typeof clinic === 'string' ? clinic : clinic.address)}`
                              }
                              target="_blank"
                              rel="noopener noreferrer"
                            >
                              <img src={googlePoint} alt="Map Icon" className="map-icon" />
                              <span>
                                {typeof clinic === 'string' ? clinic : clinic.address}
                                {clinic.distance && ` (${clinic.distance.toFixed(1)} km away)`}
                              </span>
                            </a>
                          </div>
                        ))
                      ) : (
                        <div className="clinic">
                          <span>Clinic information not available</span>
                        </div>
                      )}
                    </div>
                    {doctor.link && (
                      <a href={doctor.link} target="_blank" rel="noopener noreferrer" className="doctor-link">
                        View Profile
                      </a>
                    )}
                  </div>
                ))}
              </div>
            )}
          </React.Fragment>
        ))}
        
        {/* 👇 Auto-scroll marker */}
        <div ref={chatBottomRef} />
      </>
    );
  };
  

  const fetchRes = async () => {
    if (query || file) {
      setLoading(true);

      if (file) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('location', JSON.stringify(userLocation));

        try {
          const response = await axios.post("http://localhost:5001/api/ImageAi", formData);
          const result = response.data;
          console.log(`result from backend is ${result}`) ; 
          // Convert file to base64 before storing
          const base64Image = await convertFileToBase64(file);
          
          const dataPack = {
            imageBase64: base64Image,
            res: result.disease || "Disease might be",
            desc: result.treatment || "No treatment found",
            doctors: result.doctors || []
          };
          
          setMainRes(prevMainRes => [...prevMainRes, dataPack]);
          await saveMessage(dataPack);
        } catch (error) {
          console.error("Error processing image:", error);
        } finally {
          setLoading(false);
          setFile(null);
          const myInput = document.getElementById("myInput");
          if (myInput) myInput.disabled = false;
          setQuery("");
        }
      } else {
        // Text query handling remains the same...
        try {
          const response = await axios.post("http://localhost:5001/api/TextAi", {
            inputText: query,
            location: userLocation,
          });
          const result = response.data;
          console.log(`result from backend is ${result}`) ; 
          console.log('Result from backend:', result);
          const descText = result.treatment || "No treatment found.";
          const extraNote = result.disease
            ? "\n\n🧭 Based on your location, we’ve listed 3 recommended dermatologists below. Please consult them if your symptoms persist."
            : "";

          const dataPack = {
            query: query,
            res: result.disease || "Disease might be",
            desc: descText + extraNote,
            doctors: result.doctors || []
          };

          
          setMainRes(prev => [...prev, dataPack]);
          await saveMessage(dataPack);
          
          
        } catch (error) {
          console.error("Error processing query:", error);
        } finally {
          setLoading(false);
          setQuery('');
        }
      }
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file && file.type.startsWith("image")) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setSelectedImage(reader.result);
        setShowCropper(true);
      };
      reader.readAsDataURL(file);
    }
  };

  const confirmClearChat = async () => {
    try {
      await axios.post("http://localhost:8001/clearMessages", {
        username: globalUsername
      });
      setMainRes([]);
      setShowConfirm(false);
    } catch (err) {
      console.error("Failed to clear messages:", err);
      setShowConfirm(false);
    }
  };

  const handleCroppedFileUpload = (croppedResult) => {
    if (!croppedResult) return;
  
    setFile(croppedResult.file);
    setFilePreview(croppedResult.base64);
    setQuery("Image Uploaded");
  
    // ✅ Trigger backend analysis immediately
    setTimeout(() => {
      fetchRes();
    }, 100); // slight delay to ensure state updates before fetchRes
  };
  
  
        
  let recognition = null;

if ("webkitSpeechRecognition" in window) {
  recognition = new window.webkitSpeechRecognition();
  recognition.continuous = true;
  recognition.interimResults = true;
  recognition.lang = "en-US";
} else {
  console.error("Speech recognition is not supported in this browser.");
}
  useEffect(()=>{
    if(!recognition)return;
    recognition.onresult = (event) => {
      console.log(event.results[0][0].transcript);
      setQuery(event.results[0][0].transcript)
    };

    recognition.onerror = (event) => {
      console.error("Speech recognition error:", event.error);
    };
    return () => {
      recognition.stop(); 
    };
  },[recognition])
      const handleDrop = (e) => {
        e.preventDefault();
        const droppedFile = e.dataTransfer.files[0];
        if (droppedFile && droppedFile.type.startsWith("image/")) {
          handleFileChange({ target: { files: [droppedFile] } });
        }
      };
      
      const handlePaste = (e) => {
        const items = e.clipboardData.items;
        for (const item of items) {
          if (item.type.indexOf("image") !== -1) {
            const file = item.getAsFile();
            handleFileChange({ target: { files: [file] } });
          }
        }
      };
  
      return (
        <>
          {globalUsername ? (
            <div
              className="app-area"
              onDrop={handleDrop}
              onDragOver={(e) => e.preventDefault()}
              onPaste={handlePaste}
            >
              {showCropper && selectedImage && (
                <>
                  <div className="modal-backdrop"></div>
                  <CropImageModal
                    image={selectedImage}
                    onCropComplete={(croppedFile) => {
                      setShowCropper(false);
                      setSelectedImage(null);
                      handleCroppedFileUpload(croppedFile);
                    }}
                    onCancel={() => {
                      setShowCropper(false);
                      setSelectedImage(null);
                    }}
                  />
                </>
              )}
      
              {mainRes.length === 0 ? (
                <h1 className="starting-page">
                  Hello, <span className="user-style">{globalUsername}</span>
                </h1>
              ) : (
                <div className="op">{renderFunc()}</div>
              )}
      
              <div className="search-bar-container">
                <div className="upload-area">
                  <label htmlFor="file-upload" className="custom-upload-button">
                    <img src={imgIcon} className="image-icon" alt="" />
                  </label>
                  <input
                    id="file-upload"
                    type="file"
                    accept="image/*"
                    onChange={handleFileChange}
                    className="upload-input"
                  />
                  <div onClick={() => setShowConfirm(true)} className="mic-area" title="Clear Chat">
                    <Trash size={20} />
                  </div>
                </div>
      
                <textarea
                  className="query-searcher"
                  id="myInput"
                  onChange={(e) => setQuery(e.target.value)}
                  value={query}
                  placeholder="Message VisageCheck"
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      if (!loading && (file || query.trim())) {
                        fetchRes();
                      }
                    }
                  }}
                />
      
                <p className="check-txt">
                  VisageCheck can make mistakes. Please consult with an actual doctor.
                </p>
      
                <button
                  onClick={fetchRes}
                  disabled={loading || (!file && !query)}
                  className="search-button"
                >
                  {loading ? (
                    <div className="loading-spinner">
                      <div className="spinner"></div>
                    </div>
                  ) : (
                    '🔍'
                  )}
                </button>
              </div>
            </div>
          ) : (
            <div className="not-logged-in">
              <h2>🚫 You are not logged in</h2>
              <p>
                Please <a href="/login" className="login-link">log in here</a> to use VisageCheck AI.
              </p>
            </div>
          )}
      
          {/* Clear Chat Modal */}
          {showConfirm && (
            <ConfirmDialog
              title="Clear Chat?"
              message="Are you sure you want to delete all chat history? This cannot be undone."
              onConfirm={confirmClearChat}
              onCancel={() => setShowConfirm(false)}
            />
          )}
        </>
      );
      
    }
export default App;
