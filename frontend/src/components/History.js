import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './History.css';
import JSZip from 'jszip';
import { CSSTransition, SwitchTransition } from 'react-transition-group';

function History({ onPredictionDeleted, setMessage, setMessageType }) {
  const [predictions, setPredictions] = useState([]);
  const [selectedPrediction, setSelectedPrediction] = useState(null);
  const [predictionPreview, setPredictionPreview] = useState(null);
  const [predictionImages, setPredictionImages] = useState({});
  const [confirmDeleteId, setConfirmDeleteId] = useState(null);
  const [selectedHour, setSelectedHour] = useState('6');

  useEffect(() => {
    fetchPredictions();
  }, []);

  const fetchPredictions = async () => {
    try {
      const response = await axios.get('http://localhost:5000/predictions');
      setPredictions(response.data.predictions);
    } catch (err) {
      setMessage(err.response?.data?.error || '获取预测历史失败');
      setMessageType('error');
      setTimeout(() => setMessage(''), 1000);
    }
  };

  const handlePredictionSelect = async (predictionId) => {
    try {
      const response = await axios.get(`http://localhost:5000/predictions/${predictionId}`);
      console.log('预测详情响应:', response.data);
      setSelectedPrediction(predictionId);
      setPredictionPreview(response.data);
      
      // 获取预测结果图片
      try {
        // 提取子目录名（如TimeXer24_2）
        let subDir = '';
        if (response.data.result && response.data.result.image_files && response.data.result.image_files.length > 0) {
          const firstImg = response.data.result.image_files[0];
          subDir = firstImg.split('/')[0];
        }
        if (subDir) {
          const imagesResponse = await axios.get(`http://localhost:5000/prediction_images_by_prediction/${response.data.dataset_id}/${subDir}`);
          console.log('预测图片响应:', imagesResponse.data);
          if (imagesResponse.data.images && imagesResponse.data.images.length > 0) {
            // 拼接子目录
            setPredictionImages(prev => ({
              ...prev,
              [predictionId]: imagesResponse.data.images.map(img => `${subDir}/${img}`)
            }));
          }
        } else {
          setPredictionImages(prev => ({ ...prev, [predictionId]: [] }));
        }
      } catch (imgErr) {
        console.error('获取预测图片失败:', imgErr);
      }
    } catch (err) {
      setMessage(err.response?.data?.error || '获取预测详情失败');
      setMessageType('error');
      setTimeout(() => setMessage(''), 1000);
    }
  };

  const handleDeletePrediction = async (predictionId) => {
    try {
      const response = await axios.delete(`http://localhost:5000/predictions/${predictionId}`);
      if (response.status === 200) {
        const updatedPredictions = predictions.filter(p => p.id !== predictionId);
        setPredictions(updatedPredictions);
        if (selectedPrediction === predictionId) {
          setSelectedPrediction(null);
          setPredictionPreview(null);
          setPredictionImages({});
        }
        if (onPredictionDeleted) {
          onPredictionDeleted(predictionId);
        }
        setMessage('预测记录已删除');
        setMessageType('success');
        setTimeout(() => setMessage(''), 2000);
        setConfirmDeleteId(null);
        await fetchPredictions();
      }
    } catch (error) {
      setMessage(error.response?.data?.error || '删除失败');
      setMessageType('error');
      setTimeout(() => setMessage(''), 2000);
      setConfirmDeleteId(null);
    }
  };

  const handleExportPrediction = async (predictionId) => {
    try {
      const prediction = predictions.find(p => p.id === predictionId);
      if (!prediction || !predictionImages[predictionId]) {
        setMessage('没有可导出的预测图片');
        setMessageType('error');
        setTimeout(() => setMessage(''), 1000);
        return;
      }
      const datasetResponse = await axios.get(`http://localhost:5000/datasets/${prediction.dataset_id}`);
      const dataset = datasetResponse.data;
      const timestamp = prediction.created_at
        .replace(/[:.]/g, '-')
        .replace('T', '_')
        .replace('Z', '');
      const zip = new JSZip();
      const downloadPromises = predictionImages[predictionId].map(async (image, index) => {
        try {
          const response = await axios.get(
            `http://localhost:5000/prediction_image/${prediction.dataset_id}/${image}`,
            { responseType: 'blob' }
          );
          zip.file(`prediction_${index + 1}.png`, response.data);
        } catch (err) {
          // 忽略单个图片下载失败
        }
      });
      await Promise.all(downloadPromises);
      const content = await zip.generateAsync({ type: 'blob' });
      const url = window.URL.createObjectURL(content);
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `${dataset.filename}_${timestamp}.zip`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
      setMessage('预测结果导出成功');
      setMessageType('success');
      setTimeout(() => setMessage(''), 1000);
    } catch (err) {
      setMessage(err.response?.data?.error || '导出失败');
      setMessageType('error');
      setTimeout(() => setMessage(''), 1000);
    }
  };

  const getStatusText = (status) => {
    const statusMap = {
      'completed': '已完成',
      'running': '进行中',
      'error': '失败',
      'pending': '等待中'
    };
    return statusMap[status] || status;
  };

  const getStatusClass = (status) => {
    const statusClassMap = {
      'completed': 'status-completed',
      'running': 'status-running',
      'error': 'status-error',
      'pending': 'status-pending'
    };
    return statusClassMap[status] || '';
  };

  const handleCloseDetails = () => {
    setSelectedPrediction(null);
    setPredictionPreview(null);
    setPredictionImages({});
  };

  const filteredPredictions = predictions.filter(pred => {
    let params = pred.parameters;
    if (typeof params === 'string') {
      try { params = JSON.parse(params); } catch { params = {}; }
    }
    return params.model_name === `TimeXer${selectedHour}`;
  });

  const renderPredictionList = () => {
    return (
      <div className="prediction-list">
        {filteredPredictions.map(prediction => (
          <div key={prediction.id} className="prediction-item">
            <div className="prediction-info">
              <h3>{prediction.dataset_name}</h3>
              <p>模型: {formatModelName(prediction)}</p>
              <p>预测时间: {prediction.created_at}</p>
              <p className={`status ${getStatusClass(prediction.status)}`}>状态: {getStatusText(prediction.status)}</p>
            </div>
            <div className="prediction-actions">
              <button onClick={() => handlePredictionSelect(prediction.id)}>查看详情</button>
              <button onClick={() => setConfirmDeleteId(prediction.id)} className="delete-btn">删除</button>
            </div>
          </div>
        ))}
      </div>
    );
  };

  // 新增：格式化模型名
  const formatModelName = (prediction) => {
    if (!prediction || !prediction.parameters) return '未知模型';
    let modelName = '';
    let params = prediction.parameters;
    if (typeof params === 'string') {
      try {
        params = JSON.parse(params);
      } catch (e) { params = {}; }
    }
    if (params.model_name === 'TimeXer6') modelName = 'TimeXer_6h';
    else if (params.model_name === 'TimeXer12') modelName = 'TimeXer_12h';
    else if (params.model_name === 'TimeXer24') modelName = 'TimeXer_24h';
    return modelName || '未知模型';
  };

  return (
    <div className="history-container">
      <h2>预测历史记录</h2>
      
      <div style={{ marginBottom: '16px', display: 'flex', gap: '16px' }}>
        <button
          type="button"
          className={selectedHour === '6' ? 'active' : ''}
          style={{
            padding: '6px 18px',
            borderRadius: 6,
            border: selectedHour === '6' ? '2px solid #1890ff' : '1px solid #d9d9d9',
            background: selectedHour === '6' ? '#e6f7ff' : '#fff',
            fontWeight: 600,
            cursor: 'pointer',
            outline: 'none',
            transition: 'all 0.2s',
          }}
          onClick={() => setSelectedHour('6')}
        >6小时预测</button>
        <button
          type="button"
          className={selectedHour === '12' ? 'active' : ''}
          style={{
            padding: '6px 18px',
            borderRadius: 6,
            border: selectedHour === '12' ? '2px solid #1890ff' : '1px solid #d9d9d9',
            background: selectedHour === '12' ? '#e6f7ff' : '#fff',
            fontWeight: 600,
            cursor: 'pointer',
            outline: 'none',
            transition: 'all 0.2s',
          }}
          onClick={() => setSelectedHour('12')}
        >12小时预测</button>
        <button
          type="button"
          className={selectedHour === '24' ? 'active' : ''}
          style={{
            padding: '6px 18px',
            borderRadius: 6,
            border: selectedHour === '24' ? '2px solid #1890ff' : '1px solid #d9d9d9',
            background: selectedHour === '24' ? '#e6f7ff' : '#fff',
            fontWeight: 600,
            cursor: 'pointer',
            outline: 'none',
            transition: 'all 0.2s',
          }}
          onClick={() => setSelectedHour('24')}
        >24小时预测</button>
      </div>
      
      <div className="history-content">
        <div className="predictions-list">
          <h3>预测记录列表</h3>
          <SwitchTransition mode="out-in">
            <CSSTransition
              key={selectedHour}
              classNames="page-fade-apple"
              timeout={600}
              unmountOnExit
            >
              <div>
                {filteredPredictions.length === 0 ? (
                  <div className="no-predictions">暂无预测记录</div>
                ) : (
                  renderPredictionList()
                )}
              </div>
            </CSSTransition>
          </SwitchTransition>
        </div>

        <CSSTransition
          in={!!(selectedPrediction && predictionPreview)}
          timeout={300}
          classNames="details-animate"
          unmountOnExit
        >
          <div className="prediction-details">
            <h3>预测详情</h3>
            <div className="details-content">
              <h4>{predictionPreview?.dataset_name}</h4>
              <p>模型: {formatModelName(predictionPreview)}</p>
              <p>创建时间: {predictionPreview?.created_at}</p>
              <p className={`status ${getStatusClass(predictionPreview?.status)}`}>
                状态: {getStatusText(predictionPreview?.status)}
              </p>
              <div className="prediction-actions">
                <button onClick={() => handleExportPrediction(selectedPrediction)}>导出结果</button>
                <button onClick={handleCloseDetails}>收起详情</button>
              </div>
              {predictionPreview?.metrics && (
                <div className="metrics">
                  <h4>评估指标</h4>
                  <div className="metrics-grid">
                    <div className="metric-item">
                      <span className="metric-label">MSE</span>
                      <span className="metric-value">
                        {predictionPreview.metrics.mse ? predictionPreview.metrics.mse.toFixed(4) : 'N/A'}
                      </span>
                    </div>
                    <div className="metric-item">
                      <span className="metric-label">MAE</span>
                      <span className="metric-value">
                        {predictionPreview.metrics.mae ? predictionPreview.metrics.mae.toFixed(4) : 'N/A'}
                      </span>
                    </div>
                    <div className="metric-item">
                      <span className="metric-label">RMSE</span>
                      <span className="metric-value">
                        {predictionPreview.metrics.rmse ? predictionPreview.metrics.rmse.toFixed(4) : 'N/A'}
                      </span>
                    </div>
                  </div>
                </div>
              )}
              {predictionImages[selectedPrediction] && predictionImages[selectedPrediction].length > 0 && (
                <div className="prediction-images">
                  <h4>预测结果图表</h4>
                  <div className="image-grid">
                    {predictionImages[selectedPrediction].map((image, index) => {
                      const imageUrl = `http://localhost:5000/prediction_image/${predictionPreview.dataset_id}/${image}`;
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
          </div>
        </CSSTransition>
      </div>

      <CSSTransition
        in={!!confirmDeleteId}
        timeout={350}
        classNames="apple-modal"
        unmountOnExit
      >
        <div className="confirm-dialog apple-modal">
          <p>确定要删除这条预测记录吗？</p>
          <div className="confirm-actions">
            <button onClick={() => handleDeletePrediction(confirmDeleteId)}>确定</button>
            <button onClick={() => setConfirmDeleteId(null)}>取消</button>
          </div>
        </div>
      </CSSTransition>
    </div>
  );
}

export default History;

<style>
  {`
    .hour-btn.selected {
      background: #52c41a !important;
      color: #fff !important;
      border: 1px solid #52c41a !important;
      box-shadow: none !important;
    }
    .hour-btn {
      background: #fff;
      color: #333;
      border: 1px solid #d9d9d9;
      box-shadow: none;
    }
    .hour-btn:hover {
      border: 1px solid #52c41a;
      color: #52c41a;
    }
  `}
</style> 