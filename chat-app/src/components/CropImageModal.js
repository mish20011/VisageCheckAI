import React, { useState, useCallback, useEffect } from "react";
import Cropper from "react-easy-crop";
import getCroppedImg from "../utils/cropImage";

const CropImageModal = ({ image, onCropComplete, onCancel }) => {
  const [crop, setCrop] = useState({ x: 0, y: 0 });
  const [zoom, setZoom] = useState(1);
  const [croppedAreaPixels, setCroppedAreaPixels] = useState(null);
  const [imageLoaded, setImageLoaded] = useState(false);
  const [enableCrop, setEnableCrop] = useState(true); // Toggle crop mode

  const handleCropComplete = useCallback((_, croppedAreaPixels) => {
    setCroppedAreaPixels(croppedAreaPixels);
  }, []);

  const handleDone = async () => {
    try {
      if (enableCrop && croppedAreaPixels) {
        const croppedImage = await getCroppedImg(image, croppedAreaPixels);
        onCropComplete(croppedImage);
      } else {
        // Upload full image without cropping
        const file = dataURLtoFile(image, "full-image.jpg");
        const reader = new FileReader();
        reader.onloadend = () => {
          onCropComplete({ file, base64: reader.result });
        };
        reader.readAsDataURL(file);
      }
    } catch (error) {
      console.error("Cropping failed:", error);
    }
  };
  

  useEffect(() => {
    setImageLoaded(false); // Reset for new image
  }, [image]);

  return (
    <div className="crop-container">
      <div className="cropper-area">
            {image && (
        enableCrop ? (
          <Cropper
            image={image}
            crop={crop}
            zoom={zoom}
            aspect={1}
            minZoom={0.5}
            maxZoom={3}
            onCropChange={setCrop}
            onZoomChange={setZoom}
            onCropComplete={handleCropComplete}
            onMediaLoaded={() => setImageLoaded(true)}
          />
        ) : (
          <img src={image} alt="Uploaded Preview" className="full-image-preview" />
        )
      )}

      </div>

      {image && (
        <>
          <div className="controls">
            <label style={{ fontSize: "0.9rem" }}>
              <input
                type="checkbox"
                checked={enableCrop}
                onChange={() => setEnableCrop(!enableCrop)}
              />{" "}
              Enable Cropping
            </label>
          </div>

          {enableCrop && imageLoaded && (
            <div className="controls">
              <input
                type="range"
                min="1"
                max="3"
                step="0.1"
                value={zoom}
                onChange={(e) => setZoom(Number(e.target.value))}
                className="zoom-slider"
              />
            </div>
          )}

          <div className="controls">
            <button onClick={handleDone} className="btn">✅ Crop & Upload</button>
            <button onClick={onCancel} className="btn cancel">❌ Cancel</button>
          </div>
        </>
      )}
    </div>
  );
};

function dataURLtoFile(dataurl, filename) {
  const arr = dataurl.split(",");
  const mime = arr[0].match(/:(.*?);/)[1];
  const bstr = atob(arr[1]);
  let n = bstr.length;
  const u8arr = new Uint8Array(n);
  while (n--) {
    u8arr[n] = bstr.charCodeAt(n);
  }
  return new File([u8arr], filename, { type: mime });
}

export default CropImageModal;
