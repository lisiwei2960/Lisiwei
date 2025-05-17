import React from 'react';

export default function DatasetList({
  isCollapsed,
  onCollapseToggle,
  datasets,
  selectedDataset,
  handlePreviewDataset,
  handleDatasetSelect,
  handleDeleteDataset
}) {
  return (
    <div className={`dataset-section ${isCollapsed ? 'collapsed' : ''}`}>
      <div className="dataset-header" onClick={onCollapseToggle}>
        <h2>数据集列表</h2>
        <span className="collapse-icon">{isCollapsed ? '▼' : '▲'}</span>
      </div>
      {datasets.length === 0 ? (
        <div className="no-datasets"></div>
      ) : (
        <div className="dataset-content">
          <div className="dataset-list">
            {datasets.map(dataset => (
              <div key={dataset.id} className="dataset-item">
                <div className="dataset-info">
                  <span
                    style={{
                      cursor: 'pointer',
                      color: '#1890ff',
                      textDecoration: 'underline',
                      marginRight: 0,
                      transition: 'color 0.2s',
                    }}
                    title="点击预览数据集前5行"
                    onClick={() => handlePreviewDataset(dataset.id)}
                  >
                    {dataset.filename}
                  </span>
                  <span
                    style={{
                      display: 'inline-block',
                      background: '#e6f7ff',
                      color: '#1890ff',
                      borderRadius: 12,
                      fontSize: 10,
                      padding: '2px 10px',
                      marginLeft: 0,
                      fontWeight: 500,
                      lineHeight: 1.7,
                      boxShadow: '0 1px 4px #e6f7ff',
                      border: '1px solid #91d5ff',
                      verticalAlign: 'middle',
                    }}
                    className="status status-completed"
                    title="点击预览数据集前5行"
                  >
                    点击预览
                  </span>
                </div>
                <div className="dataset-actions">
                  <button 
                    onClick={() => handleDatasetSelect(dataset.id)}
                    className={selectedDataset === dataset.id ? 'active' : ''}
                  >
                    选择
                  </button>
                  <button 
                    onClick={() => handleDeleteDataset(dataset.id)}
                    className="delete-btn"
                  >
                    删除
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
} 