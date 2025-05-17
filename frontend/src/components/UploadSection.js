import React from 'react';
import styles from './UploadSection.module.css';

const UploadSection = ({
  files, setFiles, uploading, uploadProgress, handleFileChange, handleUpload, fileInputRef
}) => {
  return (
    <div className={`${styles.uploadSection} card`}>
      <h2>上传数据集</h2>
      <hr style={{margin: '18px 0 24px 0', border: 'none', borderTop: '1.5px solid #e6eaf3'}} />
      <div className={styles.datasetRequirements}>
        <span className={styles.icon}>📄</span>
        <span>
          <b>数据集要求：</b><br/>
          1. 仅支持 <b>CSV</b> 格式，文件大小不超过 <b>1000MB</b>。<br/>
          2. 必须包含以下字段：<br/>
          <div className={styles.fieldList}>
            <div><b>date</b>（时间，格式 <code>yyyy-MM-dd HH:mm:ss</code>）</div>
            <div><b>HUFL</b>、<b>MUFL</b>、<b>LUFL</b>（均为数值型）</div>
          </div>
        </span>
      </div>
      <div className={styles.uploadControls}>
        <span style={{fontWeight:600, color:'#1890ff', marginRight:12}}>选择文件</span>
        <input 
          type="file" 
          ref={fileInputRef}
          onChange={handleFileChange} 
          accept=".csv" 
          multiple 
          disabled={uploading}
        />
        <button 
          onClick={handleUpload} 
          disabled={files.length === 0 || uploading}
        >
          {uploading ? '上传中...' : '上传'}
        </button>
      </div>
      {files.length > 0 && (
        <div className={styles.selectedFiles}>
          <h3>已选择的文件：</h3>
          <ul>
            {files.map((file, index) => {
              const info = uploadProgress[file.name] || {};
              return (
                <li key={index} className={styles.selectedFileItem}>
                  {file.name} ({(file.size / 1024).toFixed(2)} KB)
                  {info.status && (
                    <div className={styles.progressCircleWrapper}>
                      <svg className={styles.progressCircle} width="32" height="32" viewBox="0 0 32 32">
                        <circle cx="16" cy="16" r="14" fill="none" stroke="#e6eaf3" strokeWidth="4" />
                        <circle
                          cx="16" cy="16" r="14" fill="none"
                          stroke={info.status === 'error' ? '#ff4d4f' : info.status === 'success' ? '#52c41a' : '#1890ff'}
                          strokeWidth="4"
                          strokeDasharray={2 * Math.PI * 14}
                          strokeDashoffset={(1 - (info.progress || 0) / 100) * 2 * Math.PI * 14}
                          strokeLinecap="round"
                          style={{ transition: 'stroke-dashoffset 0.4s cubic-bezier(.4,0,.2,1)' }}
                        />
                      </svg>
                      <span className={styles.progressText}>{info.status === 'uploading' ? `${info.progress || 0}%` : info.status === 'success' ? '✓' : info.status === 'error' ? '✗' : ''}</span>
                    </div>
                  )}
                  {info.message && (
                    <div className={styles.uploadMessage} style={{ color: info.status === 'error' ? '#ff4d4f' : '#52c41a' }}>
                      {info.message}
                    </div>
                  )}
                </li>
              );
            })}
          </ul>
        </div>
      )}
    </div>
  );
};

export default UploadSection; 