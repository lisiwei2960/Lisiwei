import React from 'react';

export default function GlobalMessage({ message, messageType, messageFadeOut, onClose, onMouseEnter, onMouseLeave }) {
  if (!message) return null;
  return (
    <div
      className={`global-message ${messageType}${messageFadeOut ? ' fade-out' : ''}`}
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
    >
      {message}
      <span className="close-btn" onClick={onClose}>×</span>
    </div>
  );
} 