import React, { useEffect, useState } from 'react';
import './PreviewModal.css';

export default function PreviewModal({ previewVisible, previewData, onClose }) {
  const [animationClass, setAnimationClass] = useState('');
  
  useEffect(() => {
    if (previewVisible) {
      // 短暂延迟以触发动画
      setTimeout(() => {
        setAnimationClass('visible');
      }, 10);
    } else {
      setAnimationClass('');
    }
  }, [previewVisible]);

  const handleClose = () => {
    setAnimationClass('');
    // 动画完成后关闭Modal
    setTimeout(() => {
      onClose();
    }, 400);
  };
  
  if (!previewVisible) return null;
  
  return (
    <div className={`preview-modal-mask ${animationClass}`}>
      <div className={`dataset-preview-modal ${animationClass}`}>
        <h3 style={{marginTop:0}}>数据集预览（前5行）</h3>
        <div className="table-container">
          <table style={{ borderCollapse: 'collapse', width: '100%', marginBottom: 16 }}>
            <thead>
              <tr>
                {['date', 'HUFL', 'MUFL', 'LUFL'].map(key => (
                  <th key={key}>{key}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {previewData && previewData.map((row, idx) => (
                <tr key={idx}>
                  {['date', 'HUFL', 'MUFL', 'LUFL'].map((key, i) => (
                    <td key={i}>{row[key]}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div style={{textAlign:'right'}}>
          <button className="close-button" onClick={handleClose}>关闭</button>
        </div>
      </div>
    </div>
  );
} 