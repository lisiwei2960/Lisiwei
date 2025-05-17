import React, { useState, useEffect, useRef, useRef as useMessageRef } from 'react';
import { Routes, Route, Link, Navigate, useLocation } from 'react-router-dom';
import axios from 'axios';
import './App.css';
import dayjs from 'dayjs';
import utc from 'dayjs/plugin/utc';
import History from './components/History';
import Intro from './components/Intro';
import LoadingSpinner from './components/LoadingSpinner';
import UploadSection from './components/UploadSection';
import { CSSTransition, SwitchTransition } from 'react-transition-group';
import DatasetList from './components/DatasetList';
import PredictionSection from './components/PredictionSection';
import GlobalMessage from './components/GlobalMessage';
import PreviewModal from './components/PreviewModal';
import NavBar from './components/NavBar';
import AuthForm from './components/AuthForm';
import CommentPage from './components/CommentPage';
import ProfilePage from './components/ProfilePage';
import UserAdminPage from './components/UserAdminPage';
dayjs.extend(utc);

// 在 App 组件外部添加 axios 拦截器（只添加一次）
if (!window._axios401InterceptorAdded) {
  axios.interceptors.response.use(
    response => response,
    error => {
      if (error.response && error.response.status === 401) {
        localStorage.removeItem('token');
        // 这里不能直接 setToken，因为在拦截器外部
        window.dispatchEvent(new Event('jwt-unauthorized'));
      }
      return Promise.reject(error);
    }
  );
  window._axios401InterceptorAdded = true;
}

function App() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [token, setToken] = useState(localStorage.getItem('token'));
  const [files, setFiles] = useState([]);
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [datasetPreview, setDatasetPreview] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [selectedPrediction, setSelectedPrediction] = useState(null);
  const [predictionPreview, setPredictionPreview] = useState(null);
  const [error, setError] = useState('');
  const [predictionParams, setPredictionParams] = useState({});
  const [currentPrediction, setCurrentPrediction] = useState(null);
  const [predictionProgress, setPredictionProgress] = useState(null);
  const [predictionImages, setPredictionImages] = useState({});
  const [uploadProgress, setUploadProgress] = useState({});
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState('');
  const [messageType, setMessageType] = useState('');
  const [confirmDeleteId, setConfirmDeleteId] = useState(null);
  const [authTab, setAuthTab] = useState('login'); // 'login' 或 'register'
  const [isDatasetListCollapsed, setIsDatasetListCollapsed] = useState(false);
  const fileInputRef = useRef(null);
  const [previewData, setPreviewData] = useState(null);
  const [previewVisible, setPreviewVisible] = useState(false);
  const [predictHour, setPredictHour] = useState('6'); // 6, 12, 24
  const [messageFadeOut, setMessageFadeOut] = useState(false);
  const messageTimeoutRef = useRef();
  const fadeOutTimeoutRef = useRef();
  const location = useLocation();
  const [isMessageHover, setIsMessageHover] = useState(false);
  const isLoggedIn = !!token;
  const [confirmDeleteDatasetId, setConfirmDeleteDatasetId] = useState(null);
  const isAdmin = localStorage.getItem('is_admin') === 'true';

  // 自动收起/展开数据集列表：为空时收起，上传后有数据时自动展开
  useEffect(() => {
    if (datasets.length === 0) {
      setIsDatasetListCollapsed(true);
    } else {
      setIsDatasetListCollapsed(false);
    }
  }, [datasets]);

  // 校验token有效性
  const checkTokenValid = async () => {
    if (!token) return false;
    try {
      // 这里假设有/user/info接口用于校验token有效性，如果没有可用/datasets
      await axios.get('http://localhost:5000/datasets');
      return true;
    } catch (err) {
      if (err.response && err.response.status === 401) {
        setToken('');
        localStorage.removeItem('token');
        setError('登录已过期，请重新登录');
      }
      return false;
    }
  };

  useEffect(() => {
    if (token) {
      axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
      // 先校验token有效性
      checkTokenValid().then(valid => {
        if (valid) {
          fetchDatasets();
        }
      });
    }
  }, [token]);

  // error 自动消失逻辑
  useEffect(() => {
    if (error) {
      const timer = setTimeout(() => setError(''), 3000);
      return () => clearTimeout(timer);
    }
  }, [error]);

  // 新增：鼠标事件处理
  const handleMessageMouseEnter = () => {
    setIsMessageHover(true);
    if (messageTimeoutRef.current) clearTimeout(messageTimeoutRef.current);
    if (fadeOutTimeoutRef.current) clearTimeout(fadeOutTimeoutRef.current);
  };
  const handleMessageMouseLeave = () => {
    setIsMessageHover(false);
    if (messageTimeoutRef.current) clearTimeout(messageTimeoutRef.current);
    if (fadeOutTimeoutRef.current) clearTimeout(fadeOutTimeoutRef.current);
    // 重新计时
    messageTimeoutRef.current = setTimeout(() => {
      setMessageFadeOut(true);
      fadeOutTimeoutRef.current = setTimeout(() => setMessage(''), 400);
    }, 1000);
  };

  // useEffect 只依赖 message
  useEffect(() => {
    if (!message) return;
    if (!isMessageHover) {
      if (messageTimeoutRef.current) clearTimeout(messageTimeoutRef.current);
      if (fadeOutTimeoutRef.current) clearTimeout(fadeOutTimeoutRef.current);
      messageTimeoutRef.current = setTimeout(() => {
        setMessageFadeOut(true);
        fadeOutTimeoutRef.current = setTimeout(() => setMessage(''), 400);
      }, 1000);
    }
    return () => {
      if (messageTimeoutRef.current) clearTimeout(messageTimeoutRef.current);
      if (fadeOutTimeoutRef.current) clearTimeout(fadeOutTimeoutRef.current);
    };
  }, [message]);

  const handleRegister = async () => {
    try {
      const response = await axios.post('http://localhost:5000/register', { username, password });
      showMessage(response.data.message);
      setTimeout(() => { showMessage('', '', 400); }, 1000);
    } catch (err) {
      setError(err.response?.data?.error || '注册失败');
    }
  };

  const handleLogin = async () => {
    try {
      const response = await axios.post('http://localhost:5000/login', { username, password });
      const token = response.data.token;
      setToken(token);
      localStorage.setItem('token', token);
      localStorage.setItem('username', response.data.username);
      localStorage.setItem('is_admin', response.data.is_admin);
      // 设置axios默认headers
      axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
      showMessage('登录成功');
      // 登录成功后立即获取数据集列表
      await fetchDatasets();
      // 新增：登录成功后刷新个人中心页面
      window.dispatchEvent(new Event('refresh-profile'));
    } catch (err) {
      setError(err.response?.data?.error || '登录失败');
      showMessage('登录失败', 'error');
      setTimeout(() => { showMessage('', '', 400); }, 1000);
    }
  };

  const fetchDatasets = async () => {
    try {
      const response = await axios.get('http://localhost:5000/datasets');
      console.log('数据集列表响应:', response.data.datasets);
      setDatasets(response.data.datasets);
    } catch (err) {
      setError(err.response?.data?.error || '获取数据集列表失败');
    }
  };

  const handleFileChange = (e) => {
    const maxSize = 1000 * 1024 * 1024; // 1000MB
    const selectedFiles = Array.from(e.target.files);
    const oversizeFile = selectedFiles.find(file => file.size > maxSize);
    if (oversizeFile) {
      setError('文件大小不能超过1000MB：' + oversizeFile.name);
      if (fileInputRef.current) fileInputRef.current.value = '';
      return;
    }
    setFiles(selectedFiles);
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
        
        // 使用 dayjs 处理北京时间
        const now = dayjs();
        const beijingTime = now.utcOffset(8);
        const formattedTime = beijingTime.format('YYYY-MM-DD HH:mm:ss.SSSSSS');
        console.log('北京时间:', formattedTime);
        formData.append('created_at', formattedTime);

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

      await fetchDatasets();
      
      if (successCount > 0) {
        showMessage(`上传完成：${successCount}个成功，${failCount}个失败`);
      }
      
      setFiles([]);
      setUploadProgress({});
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
      
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
      setPredictionProgress(null);
      showMessage('数据集选择成功');
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

      // 传递预测时长给后端
      const modelName = predictHour === '6' ? 'TimeXer6' : predictHour === '12' ? 'TimeXer12' : 'TimeXer24';
      const response = await axios.post('http://localhost:5000/predict', {
        dataset_id: selectedDataset,
        task_name: taskName,
        model_name: modelName
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
          setCurrentPrediction(null);
          // 新增：预测成功气泡提示
          showMessage('预测任务已完成');
          if (selectedDataset) {
            const predResponse = await axios.get(`http://localhost:5000/predictions`);
            const datasetPredictions = predResponse.data.predictions
              .filter(pred => pred.dataset_id === selectedDataset)
              .sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
            setPredictions(datasetPredictions);
            if (datasetPredictions.length > 0 && datasetPredictions[0].result && datasetPredictions[0].result.image_files && datasetPredictions[0].result.image_files.length > 0) {
              const modelName = datasetPredictions[0].parameters?.model_name || (typeof datasetPredictions[0].parameters === 'string' ? JSON.parse(datasetPredictions[0].parameters).model_name : '');
              await fetchPredictionImagesByPrediction(selectedDataset, modelName, datasetPredictions[0].id);
            } else {
              setPredictionImages(prev => ({ ...prev, [datasetPredictions[0]?.id]: [] }));
            }
          }
        } else if (response.data.status === 'error') {
          clearInterval(pollInterval);
          setError(response.data.message);
          setCurrentPrediction(null);
        }
      } catch (err) {
        clearInterval(pollInterval);
        setError('获取预测进度失败');
        setCurrentPrediction(null);
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
    setConfirmDeleteDatasetId(datasetId);
  };

  const confirmDeleteDataset = async () => {
    if (!confirmDeleteDatasetId) return;
    try {
      // 先删除预测结果文件夹
      await axios.delete(`http://localhost:5000/predictions/folder/${confirmDeleteDatasetId}`);
      console.log('已删除预测结果文件夹');

      // 再删除预测结果记录
      await axios.delete(`http://localhost:5000/predictions/dataset/${confirmDeleteDatasetId}`);
      console.log('已删除数据集的预测结果');

      // 最后删除数据集
      await axios.delete(`http://localhost:5000/datasets/${confirmDeleteDatasetId}`);
      setDatasets(datasets.filter(ds => ds.id !== confirmDeleteDatasetId));
      if (selectedDataset === confirmDeleteDatasetId) {
        setSelectedDataset(null);
        setDatasetPreview(null);
        setPredictionProgress(null);
        setPredictionImages({});
        setPredictions([]);
      }
      showMessage('数据集删除成功');
      setTimeout(() => { showMessage('', '', 400); }, 1000);
    } catch (err) {
      setError(err.response?.data?.error || '删除失败');
    } finally {
      setConfirmDeleteDatasetId(null);
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
        showMessage('预测结果已删除');

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
      showMessage(error.response?.data?.error || '删除失败', 'error');
    }
  };

  const handleExportPredictions = async () => {
    if (!selectedDataset || !predictions.length) {
      setError('没有可导出的预测结果');
      return;
    }

    try {
      const dataset = datasets.find(ds => ds.id === selectedDataset);
      // 直接使用预测记录的创建时间
      const timestamp = predictions[0].created_at
        .replace(/[:.]/g, '-')
        .replace('T', '_')
        .replace('Z', '');
      
      // 获取所有预测图片
      const images = predictionImages[selectedDataset] || [];
      if (images.length === 0) {
        setError('没有可导出的预测图片');
        return;
      }

      // 创建一个zip文件
      const JSZip = (await import('jszip')).default;
      const zip = new JSZip();

      // 下载所有图片并添加到zip
      const downloadPromises = images.map(async (image, index) => {
        try {
          const response = await axios.get(
            `http://localhost:5000/prediction_image/${selectedDataset}/${image}`,
            { responseType: 'blob' }
          );
          zip.file(`prediction_${index + 1}.png`, response.data);
        } catch (err) {
          console.error(`下载图片 ${image} 失败:`, err);
        }
      });

      await Promise.all(downloadPromises);

      // 生成zip文件
      const content = await zip.generateAsync({ type: 'blob' });
      
      // 创建下载链接
      const url = window.URL.createObjectURL(content);
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `${dataset.filename}_${timestamp}.zip`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);

      showMessage('预测结果导出成功');
      setTimeout(() => { showMessage('', '', 400); }, 1000);
    } catch (err) {
      console.error('导出失败:', err);
      setError(err.response?.data?.error || '导出失败');
      showMessage('导出失败', 'error');
    }
  };

  const fetchPredictionImagesByPrediction = async (datasetId, modelName, predictionId) => {
    try {
      const imagesResponse = await axios.get(`http://localhost:5000/prediction_images_by_prediction/${datasetId}/${modelName}`);
      if (imagesResponse.data.images && imagesResponse.data.images.length > 0) {
        setPredictionImages(prev => ({
          ...prev,
          [predictionId]: imagesResponse.data.images.map(img => `${modelName}/${img}`)
        }));
      } else {
        setPredictionImages(prev => ({ ...prev, [predictionId]: [] }));
      }
    } catch (imgErr) {
      setPredictionImages(prev => ({ ...prev, [predictionId]: [] }));
    }
  };

  const renderMainContent = (location) => (
    <div className="main-content">
      <NavBar onLogout={() => {
        setToken('');
        showMessage('已退出登录');
      }} isAdmin={isAdmin} />
      {location.pathname === '/' && (
        <DatasetList
          isCollapsed={isDatasetListCollapsed}
          onCollapseToggle={() => setIsDatasetListCollapsed(!isDatasetListCollapsed)}
          datasets={datasets}
          selectedDataset={selectedDataset}
          handlePreviewDataset={handlePreviewDataset}
          handleDatasetSelect={handleDatasetSelect}
          handleDeleteDataset={handleDeleteDataset}
        />
      )}
      <SwitchTransition mode="out-in">
        <CSSTransition
          key={location.pathname}
          classNames="page-fade-apple"
          timeout={600}
          unmountOnExit
        >
          <div className={`content${location.pathname === '/' ? ' with-sidebar' : ''}`} style={{ minHeight: 500 }}>
            <Routes location={location}>
              <Route path="/" element={
                <>
                  <UploadSection
                    files={files}
                    setFiles={setFiles}
                    uploading={uploading}
                    uploadProgress={uploadProgress}
                    handleFileChange={handleFileChange}
                    handleUpload={handleUpload}
                    fileInputRef={fileInputRef}
                  />
                  {datasets.length === 0 ? (
                    <div className="no-predictions">请上传数据集</div>
                  ) : datasets.length > 0 && !selectedDataset ? (
                    <div className="no-predictions">请选择数据集开始预测</div>
                  ) : selectedDataset ? <PredictionSection
                    predictHour={predictHour}
                    setPredictHour={setPredictHour}
                    selectedDataset={selectedDataset}
                    predictions={predictions}
                    predictionProgress={predictionProgress}
                    currentPrediction={currentPrediction}
                    handleExportPredictions={handleExportPredictions}
                    handlePredict={handlePredict}
                    setPredictions={setPredictions}
                    setPredictionImages={setPredictionImages}
                    predictionImages={predictionImages}
                    axios={axios}
                  /> : null}
                </>
              } />
              <Route path="/history" element={
                <History 
                  onPredictionDeleted={(predictionId) => {
                    setPredictions(prev => prev.filter(p => p.id !== predictionId));
                    if (selectedPrediction && selectedPrediction.id === predictionId) {
                      setSelectedPrediction(null);
                      setPredictionPreview(null);
                    }
                    if (selectedDataset) {
                      fetchDatasetImages(selectedDataset);
                    }
                    setCurrentPrediction(null);
                  }} 
                  setMessage={setMessage}
                  setMessageType={setMessageType}
                />
              } />
              <Route path="/intro" element={<Intro />} />
              <Route path="/comments" element={<CommentPage showMessage={showMessage} />} />
              <Route path="/profile" element={<ProfilePage showMessage={showMessage} />} />
              <Route path="/admin/users" element={isAdmin ? <UserAdminPage showMessage={showMessage} /> : <Navigate to="/" replace />} />
            </Routes>
          </div>
        </CSSTransition>
      </SwitchTransition>
    </div>
  );

  // 预览数据集前5行
  const handlePreviewDataset = async (datasetId) => {
    try {
      const token = localStorage.getItem('token');
      const response = await axios.get(`http://localhost:5000/datasets/${datasetId}/preview`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setPreviewData(response.data.preview);
      setPreviewVisible(true);
    } catch (err) {
      setError('获取数据集预览失败');
    }
  };

  // 优化setMessage逻辑，防止动画错乱
  const showMessage = (msg, type = 'success', duration = 1000) => {
    if (messageTimeoutRef.current) clearTimeout(messageTimeoutRef.current);
    if (fadeOutTimeoutRef.current) clearTimeout(fadeOutTimeoutRef.current);
    setMessageFadeOut(false);
    setMessage(msg);
    setMessageType(type);
    messageTimeoutRef.current = setTimeout(() => {
      setMessageFadeOut(true);
      fadeOutTimeoutRef.current = setTimeout(() => setMessage(''), 400);
    }, duration);
  };

  // 关闭按钮也要加动画
  const handleCloseMessage = () => {
    if (messageTimeoutRef.current) clearTimeout(messageTimeoutRef.current);
    if (fadeOutTimeoutRef.current) clearTimeout(fadeOutTimeoutRef.current);
    setMessageFadeOut(true);
    fadeOutTimeoutRef.current = setTimeout(() => setMessage(''), 400);
  };

  useEffect(() => {
    const handleRipple = (e) => {
      const button = e.currentTarget;
      // 防止多次点击残留
      const oldRipple = button.querySelector('.ripple');
      if (oldRipple) oldRipple.remove();
      const circle = document.createElement('span');
      const diameter = Math.max(button.clientWidth, button.clientHeight);
      const radius = diameter / 2;
      circle.style.width = circle.style.height = `${diameter}px`;
      circle.style.left = `${e.clientX - button.getBoundingClientRect().left - radius}px`;
      circle.style.top = `${e.clientY - button.getBoundingClientRect().top - radius}px`;
      circle.className = 'ripple';
      button.appendChild(circle);
    };
    // 只为页面上所有button自动绑定
    const addRippleToButtons = () => {
      const btns = document.querySelectorAll('button, .btn-apple, .btn');
      btns.forEach(btn => {
        // 避免重复绑定
        if (!btn._hasRipple) {
          btn.addEventListener('click', handleRipple);
          btn._hasRipple = true;
        }
      });
    };
    addRippleToButtons();
    // 监听DOM变化，动态按钮也能自动绑定
    const observer = new MutationObserver(addRippleToButtons);
    observer.observe(document.body, { childList: true, subtree: true });
    return () => {
      observer.disconnect();
      const btns = document.querySelectorAll('button, .btn-apple, .btn');
      btns.forEach(btn => {
        if (btn._hasRipple) {
          btn.removeEventListener('click', handleRipple);
          btn._hasRipple = false;
        }
      });
    };
  }, []);

  // 在 App 组件内部 useEffect 监听 401 事件
  useEffect(() => {
    const handleJwtUnauthorized = () => {
      setToken('');
    };
    window.addEventListener('jwt-unauthorized', handleJwtUnauthorized);
    return () => {
      window.removeEventListener('jwt-unauthorized', handleJwtUnauthorized);
    };
  }, []);

  return (
    <div className="app">
      {!isLoggedIn && location.pathname !== '/' && location.pathname !== '/intro' ? (
        <Navigate to="/" replace />
      ) : (
        <>
          {/* 全局消息提示 */}
          <GlobalMessage
            message={message}
            messageType={messageType}
            messageFadeOut={messageFadeOut}
            onClose={handleCloseMessage}
            onMouseEnter={handleMessageMouseEnter}
            onMouseLeave={handleMessageMouseLeave}
          />
          {/* 全局错误提示 */}
          {error && (
            <div className="global-message error">
              {error}
              <span className="close-btn" onClick={() => setError('')}>×</span>
            </div>
          )}
          {/* 预览弹窗 */}
          <PreviewModal
            previewVisible={previewVisible}
            previewData={previewData}
            onClose={() => setPreviewVisible(false)}
          />
          {/* 删除数据集确认弹窗 */}
          <CSSTransition
            in={!!confirmDeleteDatasetId}
            timeout={350}
            classNames="apple-modal"
            unmountOnExit
          >
            <div className="confirm-dialog apple-modal">
              <p>确定要删除该数据集吗？</p>
              <div className="confirm-actions">
                <button onClick={confirmDeleteDataset}>确定</button>
                <button onClick={() => setConfirmDeleteDatasetId(null)}>取消</button>
              </div>
            </div>
          </CSSTransition>
          <SwitchTransition mode="out-in">
            <CSSTransition
              key={!!token}
              classNames="page-fade-apple"
              timeout={600}
              unmountOnExit
            >
              <div>
                {!token ? <AuthForm
                  authTab={authTab}
                  setAuthTab={setAuthTab}
                  username={username}
                  setUsername={setUsername}
                  password={password}
                  setPassword={setPassword}
                  handleLogin={handleLogin}
                  handleRegister={handleRegister}
                /> : renderMainContent(location)}
              </div>
            </CSSTransition>
          </SwitchTransition>
        </>
      )}
    </div>
  );
}

export default App;