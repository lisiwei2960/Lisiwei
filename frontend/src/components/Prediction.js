import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Card, Button, Select, message, Progress, Table, Space, Modal, Tag } from 'antd';
import { Line } from '@ant-design/charts';

const { Option } = Select;

// 配置axios默认值
const api = axios.create({
  baseURL: 'http://localhost:5000',
  headers: {
    'Content-Type': 'application/json'
  },
  withCredentials: true
});

// 添加请求拦截器
api.interceptors.request.use(
  config => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  error => {
    return Promise.reject(error);
  }
);

// 添加响应拦截器
api.interceptors.response.use(
  response => response,
  error => {
    if (error.response) {
      console.error('API错误:', error.response.data);
      if (error.response.status === 401) {
        // 处理未授权错误
        message.error('登录已过期，请重新登录');
        localStorage.removeItem('token');
        window.location.href = '/login';
      } else {
        message.error(error.response.data.error || '请求失败');
      }
    } else {
      console.error('网络错误:', error);
      message.error('网络错误，请检查网络连接');
    }
    return Promise.reject(error);
  }
);

const Prediction = () => {
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [loading, setLoading] = useState(false);
  const [predictions, setPredictions] = useState([]);
  const [predictionProgress, setPredictionProgress] = useState({});
  const [selectedPrediction, setSelectedPrediction] = useState(null);
  const [previewVisible, setPreviewVisible] = useState(false);
  const [previewData, setPreviewData] = useState(null);
  const [selectedHour, setSelectedHour] = useState('24');
  const [imageFiles, setImageFiles] = useState([]);

  // 获取数据集列表
  useEffect(() => {
    const fetchDatasets = async () => {
      try {
        const response = await api.get('/api/datasets');
        console.log('获取到的数据集列表:', response.data);
        if (response.data.datasets) {
          setDatasets(response.data.datasets);
        }
      } catch (error) {
        console.error('获取数据集列表失败:', error);
        message.error('获取数据集列表失败');
      }
    };

    fetchDatasets();
  }, []);

  // 获取预测历史
  const fetchPredictions = async () => {
    try {
      const response = await api.get('/api/predictions');
      console.log('Predictions response:', response.data);
      if (response.data.predictions) {
        setPredictions(response.data.predictions);
      } else {
        console.error('预测历史数据格式不正确:', response.data);
        message.error('获取预测历史失败：数据格式不正确');
      }
    } catch (error) {
      console.error('Error fetching predictions:', error);
    }
  };

  // 初始加载预测历史
  useEffect(() => {
    fetchPredictions();
  }, []);

  // 处理预测请求
  const handlePredict = async () => {
    if (!selectedDataset) {
      message.warning('请选择数据集');
      return;
    }

    setLoading(true);
    try {
      console.log('发送预测请求:', {
        dataset_id: selectedDataset,
        model_name: `TimeXer${selectedHour}`  // 根据选择的小时动态选择模型
      });

      const response = await api.post('/api/predict', {
        dataset_id: selectedDataset,
        model_name: `TimeXer${selectedHour}`  // 根据选择的小时动态选择模型
      });

      console.log('预测请求响应:', response.data);
      if (response.data.prediction_id) {
        message.success('预测任务已启动');
        pollPredictionProgress(response.data.prediction_id);
      } else {
        message.error('启动预测任务失败：未获取到预测ID');
      }
    } catch (error) {
      console.error('预测请求失败:', error);
    } finally {
      setLoading(false);
    }
  };

  // 轮询预测进度
  const pollPredictionProgress = async (predictionId) => {
    const pollInterval = setInterval(async () => {
      try {
        console.log('获取预测进度:', predictionId);
        const response = await api.get(`/api/prediction_progress/${predictionId}`);
        console.log('预测进度响应:', response.data);
        
        if (response.data) {
          const progress = response.data;
          setPredictionProgress(prev => ({
            ...prev,
            [predictionId]: progress
          }));

          if (progress.status === 'completed' || progress.status === 'error') {
            clearInterval(pollInterval);
            // 更新预测列表
            fetchPredictions();
          }
        } else {
          console.error('预测进度数据格式不正确:', response.data);
          clearInterval(pollInterval);
        }
      } catch (error) {
        console.error('获取预测进度失败:', error);
        clearInterval(pollInterval);
      }
    }, 2000);
  };

  // 处理删除预测
  const handleDeletePrediction = async (predictionId) => {
    try {
      await api.delete(`/predictions/${predictionId}`);
      message.success('预测记录已删除');
      fetchPredictions(); // 重新获取预测列表
    } catch (error) {
      console.error('删除预测失败:', error);
      message.error('删除预测失败');
    }
  };

  useEffect(() => {
    if (selectedDataset && selectedHour) {
      axios.get(`http://localhost:5000/prediction_images_by_prediction/${selectedDataset}/TimeXer${selectedHour}`)
        .then(res => {
          setImageFiles(res.data.images || []);
        });
    } else {
      setImageFiles([]);
    }
  }, [selectedDataset, selectedHour]);

  const renderPredictionResult = (prediction) => {
    if (!prediction.result) return null;
    
    const result = JSON.parse(prediction.result);
    if (result.status === 'error') {
      return <div style={{ color: 'red' }}>{result.error}</div>;
    }

    return (
      <div>
        {result.metrics && (
          <div style={{ marginBottom: 16 }}>
            <h3>预测指标</h3>
            <Table
              dataSource={[
                { key: 'mse', name: 'MSE', value: result.metrics.mse?.toFixed(4) },
                { key: 'mae', name: 'MAE', value: result.metrics.mae?.toFixed(4) },
                { key: 'rmse', name: 'RMSE', value: result.metrics.rmse?.toFixed(4) },
                { key: 'mape', name: 'MAPE', value: result.metrics.mape?.toFixed(4) },
                { key: 'mspe', name: 'MSPE', value: result.metrics.mspe?.toFixed(4) }
              ]}
              columns={[
                { title: '指标', dataIndex: 'name', key: 'name' },
                { title: '值', dataIndex: 'value', key: 'value' }
              ]}
              pagination={false}
              size="small"
            />
          </div>
        )}

        {result.prediction_data && (
          <div style={{ marginBottom: '16px' }}>
            <h3>预测结果</h3>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '16px' }}>
              <div style={{ width: '45%' }}>
                <h4>预测值</h4>
                <Table
                  dataSource={result.prediction_data.predictions.map((row, index) => ({
                    key: index,
                    index: index + 1,
                    value: row[0]
                  }))}
                  columns={[
                    { title: '序号', dataIndex: 'index', key: 'index' },
                    { title: '预测值', dataIndex: 'value', key: 'value', render: val => val.toFixed(4) }
                  ]}
                  scroll={{ y: 400 }}
                  size="small"
                />
              </div>
              <div style={{ width: '45%' }}>
                <h4>真实值</h4>
                <Table
                  dataSource={result.prediction_data.groundtruth.map((row, index) => ({
                    key: index,
                    index: index + 1,
                    value: row[0]
                  }))}
                  columns={[
                    { title: '序号', dataIndex: 'index', key: 'index' },
                    { title: '真实值', dataIndex: 'value', key: 'value', render: val => val.toFixed(4) }
                  ]}
                  scroll={{ y: 400 }}
                  size="small"
                />
              </div>
            </div>
          </div>
        )}

        {(imageFiles.length > 0 ? imageFiles : result.image_files)?.length > 0 && (
          <div>
            <h3>预测结果图表</h3>
            <div style={{
              width: '100%',
              overflowX: 'auto',
              whiteSpace: 'nowrap',
              paddingBottom: '10px'
            }}>
              <div style={{
                display: 'flex',
                gap: '10px',
                padding: '10px'
              }}>
                {(imageFiles.length > 0 ? imageFiles : result.image_files)
                  .filter(image => image.startsWith(`TimeXer${selectedHour}/`) || (!image.includes('TimeXer') && imageFiles.length > 0))
                  .map((image, index) => (
                    <img
                      key={index}
                      src={
                        image.startsWith('TimeXer')
                          ? `http://localhost:5000/prediction_image/${selectedDataset}/${image}`
                          : `http://localhost:5000/prediction_image/${selectedDataset}/TimeXer${selectedHour}/${image}`
                      }
                      alt={`预测结果 ${index + 1}`}
                      style={{
                        width: '300px',
                        height: 'auto',
                        borderRadius: '4px',
                        boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
                        cursor: 'pointer',
                        flexShrink: 0
                      }}
                      onClick={() => window.open(
                        image.startsWith('TimeXer')
                          ? `http://localhost:5000/prediction_image/${selectedDataset}/${image}`
                          : `http://localhost:5000/prediction_image/${selectedDataset}/TimeXer${selectedHour}/${image}`
                      )}
                      onError={(e) => {
                        console.error('图片加载失败:', image);
                        e.target.style.display = 'none';
                      }}
                    />
                  ))}
              </div>
            </div>
            <style>
              {`
                /* 美化滚动条 */
                div::-webkit-scrollbar {
                  height: 8px;
                }
                div::-webkit-scrollbar-track {
                  background: #f1f1f1;
                  border-radius: 16px;
                }
                div::-webkit-scrollbar-thumb {
                  background: #888;
                  border-radius: 16px;
                }
                div::-webkit-scrollbar-thumb:hover {
                  background: #555;
                }
              `}
            </style>
          </div>
        )}
      </div>
    );
  };

  // 渲染前筛选预测结果
  const filteredPredictions = predictions.filter(pred => {
    let params = pred.parameters;
    if (typeof params === 'string') {
      try { params = JSON.parse(params); } catch { params = {}; }
    }
    return params.model_name === `TimeXer${selectedHour}`;
  });

  // 定义表格列
  const columns = [
    {
      title: '数据集',
      dataIndex: 'dataset_name',
      key: 'dataset_name',
    },
    {
      title: '模型',
      dataIndex: 'model',
      key: 'model',
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status) => {
        let color = 'default';
        let text = status;
        
        switch (status) {
          case 'completed':
            color = 'success';
            text = '完成';
            break;
          case 'running':
            color = 'processing';
            text = '运行中';
            break;
          case 'error':
            color = 'error';
            text = '错误';
            break;
          default:
            text = '未知';
        }
        
        return <Tag color={color}>{text}</Tag>;
      }
    },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      key: 'created_at',
    },
    {
      title: '操作',
      key: 'action',
      render: (_, record) => (
        <Space size="middle">
          <Button 
            type="link" 
            onClick={() => {
              setSelectedPrediction(record);
              setPreviewVisible(true);
              console.log('查看预测结果:', record); // 添加调试信息
            }}
          >
            查看结果
          </Button>
          <Button 
            type="link" 
            danger
            onClick={() => handleDeletePrediction(record.id)}
          >
            删除
          </Button>
        </Space>
      ),
    },
  ];

  return (
    <div style={{ padding: '24px' }}>
      <Card title="预测" style={{ marginBottom: '24px' }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '16px', marginBottom: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
            <Select
              style={{ width: 200 }}
              placeholder="选择数据集"
              onChange={setSelectedDataset}
              value={selectedDataset}
            >
              {datasets.map(dataset => (
                <Option key={dataset.id} value={dataset.id}>{dataset.filename}</Option>
              ))}
            </Select>
            <Button 
              type="primary" 
              onClick={handlePredict}
              loading={loading}
              disabled={!selectedDataset}
            >
              开始预测
            </Button>
          </div>
          
          <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
            <span style={{ marginRight: '8px' }}>预测时长：</span>
            <Select
              style={{ width: 120 }}
              value={selectedHour}
              onChange={setSelectedHour}
            >
              <Option value="6">6小时</Option>
              <Option value="12">12小时</Option>
              <Option value="24">24小时</Option>
            </Select>
          </div>
          
          <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
            <span style={{ marginRight: '8px' }}>预测参数设置：</span>
            <Select
              style={{ width: 120 }}
              defaultValue="96"
              placeholder="序列长度"
            >
              <Option value="96">96</Option>
              <Option value="192">192</Option>
              <Option value="336">336</Option>
            </Select>
            <Select
              style={{ width: 120 }}
              defaultValue="48"
              placeholder="标签长度"
            >
              <Option value="48">48</Option>
              <Option value="96">96</Option>
              <Option value="192">192</Option>
            </Select>
            <Select
              style={{ width: 120 }}
              defaultValue="6"
              placeholder="预测长度"
              disabled
            >
              <Option value="6">6小时</Option>
            </Select>
          </div>
        </div>
      </Card>

      <Card title="预测历史">
        <Table 
          columns={columns} 
          dataSource={filteredPredictions}
          rowKey="id"
          pagination={{ pageSize: 10 }}
        />
      </Card>

      <Modal
        title="预测详情"
        open={previewVisible}
        onCancel={() => setPreviewVisible(false)}
        width={1200}
        style={{ top: 20 }}
        bodyStyle={{ maxHeight: 'calc(100vh - 200px)', overflowY: 'auto' }}
        footer={null}
      >
        {selectedPrediction && (
          <div style={{ padding: '20px' }}>
            {renderPredictionResult(selectedPrediction)}
          </div>
        )}
      </Modal>

      <style>
        {`
          .ant-modal-body {
            overflow-x: hidden;
          }
          .ant-modal-body::-webkit-scrollbar {
            width: 6px;
          }
          .ant-modal-body::-webkit-scrollbar-track {
            background: #f1f1f1;
          }
          .ant-modal-body::-webkit-scrollbar-thumb {
            background: #888;
            border-radius: 3px;
          }
          .ant-modal-body::-webkit-scrollbar-thumb:hover {
            background: #555;
          }
        `}
      </style>
    </div>
  );
};

export default Prediction;