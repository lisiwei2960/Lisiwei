import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './ProfilePage.css';
import LoadingSpinner from './LoadingSpinner';

function ProfilePage({ showMessage }) {
  const [userInfo, setUserInfo] = useState({
    username: '',
    email: '',
    role: '',
    createdAt: '',
    lastLogin: ''
  });
  const [loading, setLoading] = useState(true);
  const [editing, setEditing] = useState(false);
  const [editForm, setEditForm] = useState({
    email: '',
    password: '',
    newPassword: '',
    confirmPassword: ''
  });
  const [stats, setStats] = useState({
    datasetsCount: 0,
    predictionsCount: 0
  });

  useEffect(() => {
    fetchUserInfo();
    fetchUserStats();
    // 监听刷新事件
    const refresh = () => {
      fetchUserInfo();
      fetchUserStats();
    };
    window.addEventListener('refresh-profile', refresh);
    return () => window.removeEventListener('refresh-profile', refresh);
  }, []);

  const fetchUserInfo = async () => {
    try {
      setLoading(true);
      const response = await axios.get('http://localhost:5000/user/info');
      setUserInfo({
        username: response.data.username || '',
        email: response.data.email || '',
        role: response.data.role || 'user',
        createdAt: response.data.createdAt || '',
        lastLogin: response.data.lastLogin || ''
      });
      setEditForm({
        ...editForm,
        email: response.data.email || ''
      });
    } catch (err) {
      showMessage && showMessage('获取用户信息失败', 'error');
      console.error('获取用户信息错误:', err);
    } finally {
      setLoading(false);
    }
  };

  const fetchUserStats = async () => {
    try {
      const response = await axios.get('http://localhost:5000/user/stats');
      setStats(response.data);
    } catch (err) {
      showMessage && showMessage('获取用户统计信息失败', 'error');
      console.error('获取用户统计信息错误:', err);
    }
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setEditForm({
      ...editForm,
      [name]: value
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    // 验证密码
    if (editForm.newPassword && editForm.newPassword !== editForm.confirmPassword) {
      showMessage && showMessage('两次输入的新密码不一致', 'error');
      return;
    }

    try {
      const updateData = {
        email: editForm.email
      };

      if (editForm.newPassword) {
        updateData.oldPassword = editForm.password;
        updateData.newPassword = editForm.newPassword;
      }

      await axios.put('http://localhost:5000/user/update', updateData);
      showMessage && showMessage('个人信息更新成功', 'success');
      setEditing(false);
      fetchUserInfo(); // 重新获取用户信息
    } catch (err) {
      showMessage && showMessage(err.response?.data?.error || '更新个人信息失败', 'error');
    }
  };

  if (loading) {
    return <div className="profile-container"><LoadingSpinner /></div>;
  }

  return (
    <div className="profile-container">
      <h2 className="profile-title">个人中心</h2>
      
      <div className="profile-stats">
        <div className="stat-box">
          <div className="stat-value">{stats.datasetsCount}</div>
          <div className="stat-label">数据集数量</div>
        </div>
        <div className="stat-box">
          <div className="stat-value">{stats.predictionsCount}</div>
          <div className="stat-label">预测任务数量</div>
        </div>
      </div>

      <div className="profile-card">
        <div className="profile-header">
          <h3>基本信息</h3>
          {!editing && (
            <button 
              className="edit-button" 
              onClick={() => setEditing(true)}
            >
              编辑
            </button>
          )}
        </div>

        {editing ? (
          <form onSubmit={handleSubmit} className="profile-form">
            <div className="form-group">
              <label>用户名</label>
              <input 
                type="text" 
                value={userInfo.username} 
                disabled 
              />
            </div>
            <div className="form-group">
              <label>邮箱</label>
              <input 
                type="email" 
                name="email" 
                value={editForm.email} 
                onChange={handleInputChange} 
              />
            </div>
            <div className="form-group">
              <label>当前密码</label>
              <input 
                type="password" 
                name="password" 
                value={editForm.password} 
                onChange={handleInputChange} 
                placeholder="输入当前密码以验证身份"
              />
            </div>
            <div className="form-group">
              <label>新密码</label>
              <input 
                type="password" 
                name="newPassword" 
                value={editForm.newPassword} 
                onChange={handleInputChange} 
                placeholder="留空表示不修改密码"
              />
            </div>
            <div className="form-group">
              <label>确认新密码</label>
              <input 
                type="password" 
                name="confirmPassword" 
                value={editForm.confirmPassword} 
                onChange={handleInputChange} 
                placeholder="再次输入新密码"
              />
            </div>
            
            <div className="form-buttons">
              <button type="button" className="cancel-button" onClick={() => setEditing(false)}>
                取消
              </button>
              <button type="submit" className="save-button">
                保存
              </button>
            </div>
          </form>
        ) : (
          <div className="profile-info">
            <div className="info-row">
              <span className="info-label">用户名:</span>
              <span className="info-value">{userInfo.username}</span>
            </div>
            <div className="info-row">
              <span className="info-label">邮箱:</span>
              <span className="info-value">{userInfo.email || '未设置'}</span>
            </div>
            <div className="info-row">
              <span className="info-label">角色:</span>
              <span className="info-value">{userInfo.role === 'admin' ? '管理员' : '普通用户'}</span>
            </div>
            <div className="info-row">
              <span className="info-label">注册时间:</span>
              <span className="info-value">{userInfo.createdAt ? new Date(userInfo.createdAt).toLocaleString() : '未知'}</span>
            </div>
            <div className="info-row">
              <span className="info-label">最后登录:</span>
              <span className="info-value">{userInfo.lastLogin ? new Date(userInfo.lastLogin).toLocaleString() : '未知'}</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default ProfilePage; 