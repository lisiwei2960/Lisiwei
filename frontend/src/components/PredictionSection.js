import React from 'react';

export default function PredictionSection({
  predictHour,
  setPredictHour,
  selectedDataset,
  predictions,
  predictionProgress,
  currentPrediction,
  handleExportPredictions,
  handlePredict,
  fetchPredictionImagesByPrediction,
  setPredictions,
  setPredictionImages,
  predictionImages,
  axios
}) {
  // 修复：只判断当前时长的预测结果
  const modelName = predictHour === '6' ? 'TimeXer6' : predictHour === '12' ? 'TimeXer12' : 'TimeXer24';
  const currentHourPredictions = predictions.filter(pred => {
    if (typeof pred.parameters === 'string') {
      try {
        return JSON.parse(pred.parameters).model_name === modelName;
      } catch {
        return false;
      }
    }
    return pred.parameters?.model_name === modelName;
  });

  return (
    <div className="prediction-section card">
      <h2>预测设置</h2>
      {/* 预测时长三选一按钮 */}
      <div style={{marginBottom: 16, display: 'flex', gap: 16}}>
        <button
          type="button"
          className={predictHour === '6' ? 'active' : ''}
          style={{
            padding: '6px 18px',
            borderRadius: 6,
            border: predictHour === '6' ? '2px solid #1890ff' : '1px solid #d9d9d9',
            background: predictHour === '6' ? '#e6f7ff' : '#fff',
            fontWeight: 600,
            cursor: 'pointer',
            outline: 'none',
            transition: 'all 0.2s',
          }}
          onClick={() => setPredictHour('6')}
        >6小时预测</button>
        <button
          type="button"
          className={predictHour === '12' ? 'active' : ''}
          style={{
            padding: '6px 18px',
            borderRadius: 6,
            border: predictHour === '12' ? '2px solid #1890ff' : '1px solid #d9d9d9',
            background: predictHour === '12' ? '#e6f7ff' : '#fff',
            fontWeight: 600,
            cursor: 'pointer',
            outline: 'none',
            transition: 'all 0.2s',
          }}
          onClick={() => setPredictHour('12')}
        >12小时预测</button>
        <button
          type="button"
          className={predictHour === '24' ? 'active' : ''}
          style={{
            padding: '6px 18px',
            borderRadius: 6,
            border: predictHour === '24' ? '2px solid #1890ff' : '1px solid #d9d9d9',
            background: predictHour === '24' ? '#e6f7ff' : '#fff',
            fontWeight: 600,
            cursor: 'pointer',
            outline: 'none',
            transition: 'all 0.2s',
          }}
          onClick={() => setPredictHour('24')}
        >24小时预测</button>
      </div>
      {selectedDataset && (
        <div className="prediction-form">
          {currentHourPredictions.length > 0 ? (
            <button 
              onClick={handleExportPredictions}
              className="export-btn"
            >
              导出预测结果
            </button>
          ) : (
            <button 
              onClick={handlePredict} 
              disabled={currentPrediction}
            >
              开始预测
            </button>
          )}
        </div>
      )}
      {predictionProgress && predictionProgress.status !== 'completed' && (
        <div className="prediction-progress">
          <h3>预测进度</h3>
          <p>状态: {predictionProgress.status}</p>
          {predictionProgress.progress && (
            <div className="progress-bar">
              <div 
                className="progress-fill"
                style={{ width: `${predictionProgress.progress}%` }}
              />
            </div>
          )}
        </div>
      )}
      {selectedDataset && predictions.length > 0 && (
        <div className="prediction-results">
          {predictions[0].metrics && (
            <div className="metrics">
              <h4>评估指标</h4>
              <div className="metrics-grid">
                <div className="metric-item">
                  <span className="metric-label">MSE</span>
                  <span className="metric-value">
                    {predictions[0].metrics.mse ? predictions[0].metrics.mse.toFixed(4) : 'N/A'}
                  </span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">MAE</span>
                  <span className="metric-value">
                    {predictions[0].metrics.mae ? predictions[0].metrics.mae.toFixed(4) : 'N/A'}
                  </span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">RMSE</span>
                  <span className="metric-value">
                    {predictions[0].metrics.rmse ? predictions[0].metrics.rmse.toFixed(4) : 'N/A'}
                  </span>
                </div>
              </div>
            </div>
          )}
          {predictionImages[selectedDataset] && predictionImages[selectedDataset].length > 0 && (
            <div className="prediction-images">
              <h4>预测结果图表</h4>
              <div className="image-grid">
                {predictionImages[selectedDataset].map((image, index) => {
                  const imageUrl = `http://localhost:5000/prediction_image/${selectedDataset}/${image}`;
                  return (
                    <div key={index} className="image-container">
                      <img 
                        src={imageUrl}
                        alt={`预测结果图 ${index + 1}`}
                        onClick={() => window.open(imageUrl)}
                        onError={(e) => {
                          console.error(`图片加载失败: ${imageUrl}`);
                          e.target.style.display = 'none';
                        }}
                      />
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      )}
      {selectedDataset && (!predictions.length || !predictionImages[selectedDataset] || predictionImages[selectedDataset].length === 0) && (
        <div className="no-predictions">
          暂无预测结果，请先进行预测
        </div>
      )}
    </div>
  );
} 