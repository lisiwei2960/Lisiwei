import React from 'react';
import './LoadingSpinner.css';

const LoadingSpinner = () => (
  <div className="loading-spinner-overlay">
    <div className="loading-spinner">
      <div className="spinner"></div>
      <div className="loading-text">加载中...</div>
    </div>
  </div>
);

export default LoadingSpinner; 