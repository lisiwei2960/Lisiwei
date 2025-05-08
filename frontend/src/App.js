import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';
import dayjs from 'dayjs';
import utc from 'dayjs/plugin/utc';
dayjs.extend(utc);

function App() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [token, setToken] = useState('');
  const [files, setFiles] = useState([]);
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [datasetPreview, setDatasetPreview] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [selectedPrediction, setSelectedPrediction] = useState(null);
  const [predictionPreview, setPredictionPreview] = useState(null);
  const [error, setError] = useState('');
  const [predictionParams, setPredictionParams] = useState({
    task_name: 'prediction_task',
    model: 'TimesNet',
    seq_len: 96,
    label_len: 48,
    pred_len: 6,
    train_epochs: 1,
    batch_size: 32
  });
  const [currentPrediction, setCurrentPrediction] = useState(null);
  const [predictionProgress, setPredictionProgress] = useState(null);
  const [predictionImages, setPredictionImages] = useState({});
  const [uploadProgress, setUploadProgress] = useState({});
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState('');
  const [messageType, setMessageType] = useState('info'); // 'info', 'success', 'error'
  const [confirmDeleteId, setConfirmDeleteId] = useState(null);
  const [authTab, setAuthTab] = useState('login'); // 'login' 或 'register'

  // 设置axios默认headers
  useEffect(() => {
    if (token) {
      axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
      fetchDatasets();
    }
  }, [token]);

  const handleRegister = async () => {
    try {
      const response = await axios.post('http://localhost:5000/register', { username, password });
      setMessage(response.data.message);
      setMessageType('success');
      setTimeout(() => setMessage(''), 3000);
    } catch (err) {
      setError(err.response?.data?.error || '注册失败');
    }
  };

  const handleLogin = async () => {
    try {
      const response = await axios.post('http://localhost:5000/login', { username, password });
      setToken(response.data.token);
      setMessage('登录成功');
      setMessageType('success');
      setTimeout(() => setMessage(''), 3000);
    } catch (err) {
      setError(err.response?.data?.error || '登录失败');
    }
  };

  const fetchDatasets = async () => {
    try {
      const response = await axios.get('http://localhost:5000/datasets');
      setDatasets(response.data.datasets);
    } catch (err) {
      setError(err.response?.data?.error || '获取数据集列表失败');
    }
  };

  const handleFileChange = (e) => {
    setFiles(Array.from(e.target.files));
  };

  const handleUpload = async () => {
    if (files.length === 0) {
      setError('请选择文件');
      return;
    }

    setUploading(true);
    setError('');
    let successCount = 0;
    let failCount = 0;

    try {
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        const formData = new FormData();
        formData.append('file', file);

        setUploadProgress(prev => ({
          ...prev,
          [file.name]: {
            progress: 0,
            status: 'uploading'
          }
        }));

        try {
          const response = await axios.post('http://localhost:5000/upload', formData, {
            onUploadProgress: (progressEvent) => {
              const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
              setUploadProgress(prev => ({
                ...prev,
                [file.name]: {
                  progress: percentCompleted,
                  status: 'uploading'
                }
              }));
            }
          });

          setUploadProgress(prev => ({
            ...prev,
            [file.name]: {
              progress: 100,
              status: 'success',
              message: response.data.message
            }
          }));
          successCount++;
        } catch (err) {
          setUploadProgress(prev => ({
            ...prev,
            [file.name]: {
              progress: 0,
              status: 'error',
              message: err.response?.data?.error || '上传失败'
            }
          }));
          failCount++;
        }
      }

      // 刷新数据集列表
      await fetchDatasets();
      
      // 显示上传结果
      if (successCount > 0) {
        setMessage(`上传完成：${successCount}个成功，${failCount}个失败`);
        setMessageType('success');
        setTimeout(() => setMessage(''), 3000);
      }
      
      // 清空文件列表和进度
      setFiles([]);
      setUploadProgress({});
      
    } catch (err) {
      setError('上传过程中发生错误');
    } finally {
      setUploading(false);
    }
  };

  const fetchDatasetImages = async (datasetId) => {
    try {
      console.log(`开始获取数据集 ${datasetId} 的预测图片`);
      const imagesResponse = await axios.get(`http://localhost:5000/prediction_images/${datasetId}`);
      console.log(`获取到图片列表:`, imagesResponse.data);
      
      if (imagesResponse.data.images && imagesResponse.data.images.length > 0) {
        setPredictionImages(prev => ({
          ...prev,
          [datasetId]: imagesResponse.data.images
        }));
        console.log(`已更新数据集 ${datasetId} 的预测图片:`, imagesResponse.data.images);
      } else {
        console.log(`数据集 ${datasetId} 没有预测图片`);
      }
    } catch (imgErr) {
      console.error(`获取数据集 ${datasetId} 的预测图片失败:`, imgErr);
    }
  };

  const handleDatasetSelect = async (datasetId) => {
    try {
      const response = await axios.get(`http://localhost:5000/datasets/${datasetId}`);
      setSelectedDataset(datasetId);
      setDatasetPreview(response.data);
      
      // 获取该数据集的预测结果
      try {
        const predResponse = await axios.get(`http://localhost:5000/predictions`);
        // 过滤出属于当前数据集的预测结果，并按时间排序
        const datasetPredictions = predResponse.data.predictions
          .filter(pred => pred.dataset_id === datasetId)
          .sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
        
        setPredictions(datasetPredictions);
        
        // 获取该数据集的预测图片
        await fetchDatasetImages(datasetId);
        
      } catch (predErr) {
        console.error('获取预测结果失败:', predErr);
        setPredictions([]);
      }
    } catch (err) {
      setError(err.response?.data?.error || '获取数据集详情失败');
      setPredictions([]);
    }
  };

  const handlePredict = async () => {
    if (!selectedDataset) {
      setError('请先选择数据集');
      return;
    }
    try {
      setError('');
      setPredictionProgress(null);
      setPredictionImages({});
      
      // 使用数据集ID作为任务名称的一部分
      const taskName = `dataset_${selectedDataset}_${Date.now()}`;
      setPredictionParams(prev => ({
        ...prev,
        task_name: taskName
      }));
      
      const response = await axios.post('http://localhost:5000/predict', {
        dataset_id: selectedDataset,
        task_name: taskName
      });
      
      setCurrentPrediction(response.data.prediction_id);
      startProgressPolling(response.data.prediction_id);
    } catch (err) {
      setError(err.response?.data?.error || '预测失败');
    }
  };

  const startProgressPolling = (predictionId) => {
    const pollInterval = setInterval(async () => {
      try {
        const response = await axios.get(`http://localhost:5000/prediction_progress/${predictionId}`);
        setPredictionProgress(response.data);
        
        if (response.data.status === 'completed') {
          clearInterval(pollInterval);
          // 预测完成后，自动更新数据
          if (selectedDataset) {
            // 获取最新的预测结果
            const predResponse = await axios.get(`http://localhost:5000/predictions`);
            // 过滤出属于当前数据集的预测结果，并按时间排序
            const datasetPredictions = predResponse.data.predictions
              .filter(pred => pred.dataset_id === selectedDataset)
              .sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
            
            setPredictions(datasetPredictions);
            console.log('已更新预测结果列表:', datasetPredictions);
            
            // 获取最新的预测图片
            await fetchDatasetImages(selectedDataset);
          }
        } else if (response.data.status === 'error') {
          clearInterval(pollInterval);
          setError(response.data.message);
        }
      } catch (err) {
        clearInterval(pollInterval);
        setError('获取预测进度失败');
      }
    }, 1000);
  };

  // 在预测完成后自动刷新数据
  useEffect(() => {
    if (predictionProgress?.status === 'completed' && selectedDataset) {
      console.log('预测完成，自动更新数据');
      // 获取最新的预测结果
      const fetchLatestData = async () => {
        try {
          const predResponse = await axios.get(`http://localhost:5000/predictions`);
          const datasetPredictions = predResponse.data.predictions
            .filter(pred => pred.dataset_id === selectedDataset)
            .sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
          
          setPredictions(datasetPredictions);
          console.log('已更新预测结果列表:', datasetPredictions);
          
          // 获取最新的预测图片
          await fetchDatasetImages(selectedDataset);
        } catch (error) {
          console.error('更新预测结果失败:', error);
        }
      };
      fetchLatestData();
    }
  }, [predictionProgress?.status, selectedDataset]);

  const handleDeleteDataset = async (datasetId) => {
    try {
      await axios.delete(`http://localhost:5000/datasets/${datasetId}`);
      setDatasets(datasets.filter(ds => ds.id !== datasetId));
      if (selectedDataset === datasetId) {
        setSelectedDataset(null);
        setDatasetPreview(null);
        setPredictionProgress(null);
        setPredictionImages({});
      }
      setMessage('数据集删除成功');
      setMessageType('success');
      setTimeout(() => setMessage(''), 3000);
      setConfirmDeleteId(null);
    } catch (err) {
      setError(err.response?.data?.error || '删除失败');
      setConfirmDeleteId(null);
    }
  };

  const handleDeletePrediction = async (predictionId) => {
    try {
      const response = await axios.delete(`http://localhost:5000/predictions/${predictionId}`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      
      if (response.status === 200) {
        // 更新预测列表
        const updatedPredictions = predictions.filter(p => p.id !== predictionId);
        setPredictions(updatedPredictions);
        
        // 如果删除的是当前选中的预测，清空预测预览
        if (selectedPrediction && selectedPrediction.id === predictionId) {
          setSelectedPrediction(null);
          setPredictionPreview(null);
        }

        // 重新获取预测图片
        if (selectedDataset) {
          await fetchDatasetImages(selectedDataset);
        }
        
        // 显示成功消息
        setMessage('预测结果已删除');
        setMessageType('success');
        setTimeout(() => setMessage(''), 3000);

        // 如果删除后没有预测结果了，清空图片显示
        if (updatedPredictions.length === 0) {
          setPredictionImages(prev => ({
            ...prev,
            [selectedDataset]: []
          }));
        }
      }
    } catch (error) {
      console.error('删除预测结果失败:', error);
      setMessage(error.response?.data?.error || '删除失败');
      setMessageType('error');
      setTimeout(() => setMessage(''), 3000);
    }
  };

  const renderPredictionImages = () => {
    const currentImages = predictionImages[selectedDataset] || [];
    console.log(`当前数据集 ${selectedDataset} 的图片:`, currentImages);
    
    if (currentImages.length === 0) {
      if (predictions.length === 0) {
        return <div className="no-predictions">暂无预测结果</div>;
      }
      return null;
    }
    
    return (
      <div className="prediction-images">
        <h3>预测结果图表</h3>
        <div className="image-grid">
          {currentImages.map((image, index) => {
            const imageUrl = `http://localhost:5000/prediction_image/${selectedDataset}/${image}`;
            console.log(`加载图片 ${index + 1}:`, imageUrl);
            
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
                  onLoad={() => console.log(`图片加载成功: ${imageUrl}`)}
                />
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  const renderUploadSection = () => (
    <div className="upload-section">
      <h2>上传数据集</h2>
      <div className="upload-controls">
        <input 
          type="file" 
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
        <div className="selected-files">
          <h3>已选择的文件：</h3>
          <ul>
            {files.map((file, index) => (
              <li key={index}>
                {file.name} ({(file.size / 1024).toFixed(2)} KB)
              </li>
            ))}
          </ul>
        </div>
      )}
      
      {Object.keys(uploadProgress).length > 0 && (
        <div className="upload-progress-list">
          <h3>上传进度：</h3>
          {Object.entries(uploadProgress).map(([fileName, info]) => (
            <div key={fileName} className={`upload-progress-item ${info.status}`}>
              <div className="file-info">
                <span className="filename">{fileName}</span>
                <span className="status">
                  {info.status === 'uploading' ? `${info.progress}%` :
                   info.status === 'success' ? '✓ 成功' :
                   info.status === 'error' ? '✗ 失败' : ''}
                </span>
              </div>
              <div className="progress-bar">
                <div 
                  className="progress-bar-fill"
                  style={{ 
                    width: `${info.progress}%`,
                    backgroundColor: info.status === 'error' ? '#ff4d4f' :
                                   info.status === 'success' ? '#52c41a' : '#1890ff'
                  }}
                ></div>
              </div>
              {info.message && (
                <div className="upload-message">
                  {info.message}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );

  // 在组件卸载时清理状态
  useEffect(() => {
    return () => {
      setPredictionImages({});
      setPredictions([]);
      setSelectedDataset(null);
      setDatasetPreview(null);
    };
  }, []);

  // 在登录状态改变时清理状态
  useEffect(() => {
    if (!token) {
      setPredictionImages({});
      setPredictions([]);
      setSelectedDataset(null);
      setDatasetPreview(null);
    }
  }, [token]);

  return (
    <>
      <div id="tech-particles-bg"></div>
      <div className="App">
        <h1>电力负荷预测系统</h1>
        
        {!token ? (
          <div className="auth-container">
            <div className="auth-form">
              <div className="auth-tabs">
                <button
                  className={authTab === 'login' ? 'active' : ''}
                  onClick={() => setAuthTab('login')}
                >登录</button>
                <button
                  className={authTab === 'register' ? 'active' : ''}
                  onClick={() => setAuthTab('register')}
                >注册</button>
              </div>
              {authTab === 'register' ? (
                <>
                  <input type="text" placeholder="用户名" value={username} onChange={(e) => setUsername(e.target.value)} />
                  <input type="password" placeholder="密码" value={password} onChange={(e) => setPassword(e.target.value)} />
                  <button onClick={handleRegister}>注册</button>
                </>
              ) : (
                <>
                  <input type="text" placeholder="用户名" value={username} onChange={(e) => setUsername(e.target.value)} />
                  <input type="password" placeholder="密码" value={password} onChange={(e) => setPassword(e.target.value)} />
                  <button onClick={handleLogin}>登录</button>
                </>
              )}
            </div>
          </div>
        ) : (
          <div className="main-container">
            {renderUploadSection()}

            <div className="datasets-section">
              <h2>我的数据集</h2>
              <div className="datasets-list">
                {datasets.map(dataset => (
                  <div 
                    key={dataset.id} 
                    className={`dataset-item ${selectedDataset === dataset.id ? 'selected' : ''}`}
                  >
                    <div className="dataset-item-content" onClick={() => handleDatasetSelect(dataset.id)}>
                      <span>{dataset.filename}</span>
                      <span className="upload-time">{dayjs(dataset.upload_time).add(8, 'hour').format('YYYY-MM-DD HH:mm:ss')}</span>
                    </div>
                    {confirmDeleteId === dataset.id ? (
                      <button
                        className="delete-button confirm"
                        onClick={e => {
                          e.stopPropagation();
                          handleDeleteDataset(dataset.id);
                        }}
                      >确认删除</button>
                    ) : (
                      <button
                        className="delete-button"
                        onClick={e => {
                          e.stopPropagation();
                          setConfirmDeleteId(dataset.id);
                        }}
                      >删除</button>
                    )}
                  </div>
                ))}
              </div>
            </div>

            {datasetPreview && (
              <div className="preview-section">
                <h2>数据集预览</h2>
                <div className="dataset-info">
                  <p>文件名：{datasetPreview.filename}</p>
                  <p>上传时间：{dayjs(datasetPreview.upload_time).add(8, 'hour').format('YYYY-MM-DD HH:mm:ss')}</p>
                  <p>总行数：{datasetPreview.row_count}</p>
                  
                  {datasetPreview.stats && (
                    <div className="dataset-stats">
                      <h3>数据集统计信息</h3>
                      <div style={{ display: 'flex', gap: 32, alignItems: 'flex-start', marginTop: 8, width: '100%' }}>
                        <div className="time-range" style={{ minWidth: 180, flexShrink: 0, margin: 0 }}>
                          <p>时间范围：</p>
                          <p>开始：{datasetPreview.stats.time_range.start}</p>
                          <p>结束：{datasetPreview.stats.time_range.end}</p>
                        </div>
                        <div className="value-ranges" style={{ flex: 1 }}>
                          <h4>数值范围统计：</h4>
                          <table className="stats-table">
                            <thead>
                              <tr>
                                <th>字段</th>
                                <th>最小值</th>
                                <th>最大值</th>
                                <th>平均值</th>
                              </tr>
                            </thead>
                            <tbody>
                              {Object.entries(datasetPreview.stats.value_ranges).map(([field, stats]) => (
                                <tr key={field}>
                                  <td>{field}</td>
                                  <td>{stats.min.toFixed(2)}</td>
                                  <td>{stats.max.toFixed(2)}</td>
                                  <td>{stats.mean.toFixed(2)}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    </div>
                  )}
                  
                  <h3>数据预览（前5行）：</h3>
                  <div className="table-container">
                    <table>
                      <thead>
                        <tr>
                          {datasetPreview.columns.map(col => (
                            <th key={col}>{col}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {datasetPreview.preview.map((row, idx) => (
                          <tr key={idx}>
                            {datasetPreview.columns.map(col => (
                              <td key={col}>
                                {col === 'date' ? row[col] : 
                                 typeof row[col] === 'number' ? row[col].toFixed(4) : row[col]}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  
                  <div className="prediction-controls">
                    <h3>预测参数设置</h3>
                    <div className="prediction-params">
                      <input
                        type="text"
                        placeholder="任务名称"
                        value={predictionParams.task_name}
                        onChange={(e) => setPredictionParams({
                          ...predictionParams,
                          task_name: e.target.value
                        })}
                      />
                    </div>
                    <button 
                      onClick={handlePredict}
                      disabled={predictionProgress?.status === 'running'}
                    >
                      {predictionProgress?.status === 'running' ? '预测中...' : '开始预测'}
                    </button>
                  </div>

                  {predictionProgress && (
                    <div className="prediction-progress">
                      <h3>预测进度</h3>
                      <div className="progress-bar">
                        <div 
                          className="progress-bar-fill" 
                          style={{ width: `${predictionProgress.progress}%` }}
                        ></div>
                      </div>
                      <div className="progress-info">
                        <p>状态: {
                          predictionProgress.status === 'running' ? '运行中' :
                          predictionProgress.status === 'completed' ? '已完成' :
                          predictionProgress.status === 'error' ? '出错' : '未知'
                        }</p>
                        <p>进度: {predictionProgress.progress}%</p>
                      </div>
                      <div className="output-log">
                        <h4>输出日志：</h4>
                        <pre className="log-content">{predictionProgress.message}</pre>
                      </div>
                    </div>
                  )}

                  {renderPredictionImages()}
                  {predictions.length > 0 && (
                    <div className="predictions-section">
                      <h2>预测结果</h2>
                      {predictions.map(prediction => (
                        <div key={prediction.id} className="prediction-item">
                          <div className="prediction-header">
                            <h3>预测时间：{prediction.created_at}</h3>
                            <button 
                              className="delete-button"
                              onClick={() => {
                                if (window.confirm('确定要删除这个预测结果吗？')) {
                                  handleDeletePrediction(prediction.id);
                                }
                              }}
                            >
                              删除
                            </button>
                          </div>
                          {prediction.result && prediction.result.metrics && (
                            <div className="metrics-table">
                              <table>
                                <thead>
                                  <tr>
                                    <th>指标</th>
                                    <th>值</th>
                                  </tr>
                                </thead>
                                <tbody>
                                  {Object.entries(prediction.result.metrics).map(([key, value]) => (
                                    <tr key={key}>
                                      <td>{key.toUpperCase()}</td>
                                      <td>{typeof value === 'number' ? value.toFixed(4) : value}</td>
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        )}

        {error && <p className="error-message">{error}</p>}
        {message && (
          <div className={`global-message ${messageType}`}>
            {message}
            <span className="close-btn" onClick={() => setMessage('')}>×</span>
          </div>
        )}
      </div>
    </>
  );
}

export default App;