import React from "react";
import "./ConfirmDialog.css";


const ConfirmDialog = ({ title, message, onConfirm, onCancel }) => {
  return (
    <div className="confirm-backdrop">
      <div className="confirm-modal">
        <h2>{title}</h2>
        <p>{message}</p>
        <div className="confirm-buttons">
          <button className="btn cancel" onClick={onCancel}>Cancel</button>
          <button className="btn confirm" onClick={onConfirm}>Confirm</button>
        </div>
      </div>
    </div>
  );
};

export default ConfirmDialog;
