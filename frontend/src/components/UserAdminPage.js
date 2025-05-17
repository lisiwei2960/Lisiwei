import React, { useEffect, useState } from 'react';
import axios from 'axios';
import './UserAdminPage.css';
import LoadingSpinner from './LoadingSpinner';

function UserAdminPage({ showMessage }) {
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [resetUserId, setResetUserId] = useState(null);
  const [resetPassword, setResetPassword] = useState('');
  const [confirmingDelete, setConfirmingDelete] = useState(null);
  const [editUserId, setEditUserId] = useState(null);
  const [editEmail, setEditEmail] = useState('');
  const [editPassword, setEditPassword] = useState('');

  useEffect(() => {
    fetchUsers();
  }, []);

  const fetchUsers = async () => {
    setLoading(true);
    try {
      const token = localStorage.getItem('token');
      const res = await axios.get('http://localhost:5000/admin/users', {
        headers: { Authorization: `Bearer ${token}` }
      });
      setUsers(res.data.users);
    } catch (err) {
      showMessage && showMessage('获取用户列表失败', 'error');
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (userId) => {
    try {
      await axios.delete(`http://localhost:5000/admin/users/${userId}`);
      showMessage && showMessage('用户已删除', 'success');
      fetchUsers();
    } catch (err) {
      showMessage && showMessage('删除用户失败', 'error');
    }
  };

  const handleResetPassword = async (userId) => {
    if (!resetPassword) {
      showMessage && showMessage('请输入新密码', 'error');
      return;
    }
    try {
      await axios.post(`http://localhost:5000/admin/users/${userId}/reset_password`, { password: resetPassword });
      showMessage && showMessage('密码已重置', 'success');
      setResetUserId(null);
      setResetPassword('');
    } catch (err) {
      showMessage && showMessage('重置密码失败', 'error');
    }
  };

  const openEditModal = (user) => {
    setEditUserId(user.id);
    setEditEmail(user.email || '');
    setEditPassword('');
  };
  const closeEditModal = () => {
    setEditUserId(null);
    setEditEmail('');
    setEditPassword('');
  };

  const handleEditSave = async () => {
    if (editEmail.trim() === '') {
      showMessage && showMessage('邮箱不能为空', 'error');
      return;
    }
    const emailPattern = /^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$/;
    if (!emailPattern.test(editEmail.trim())) {
      showMessage && showMessage('请输入有效的邮箱地址', 'error');
      return;
    }
    try {
      if (editEmail) {
        await axios.put(`http://localhost:5000/admin/users/${editUserId}/email`, { email: editEmail }, {
          headers: { Authorization: `Bearer ${localStorage.getItem('token')}` }
        });
      }
      if (editPassword) {
        await axios.post(`http://localhost:5000/admin/users/${editUserId}/reset_password`, { password: editPassword }, {
          headers: { Authorization: `Bearer ${localStorage.getItem('token')}` }
        });
      }
      showMessage && showMessage('用户信息已更新', 'success');
      closeEditModal();
      fetchUsers();
    } catch (err) {
      showMessage && showMessage(err.response?.data?.error || '更新失败', 'error');
    }
  };

  const handleEditDelete = async () => {
    try {
      await axios.delete(`http://localhost:5000/admin/users/${editUserId}`, {
        headers: { Authorization: `Bearer ${localStorage.getItem('token')}` }
      });
      showMessage && showMessage('用户已删除', 'success');
      closeEditModal();
      fetchUsers();
    } catch (err) {
      showMessage && showMessage('删除用户失败', 'error');
    }
  };

  const filteredUsers = users.filter(u =>
    u.username.toLowerCase().includes(search.toLowerCase()) ||
    (u.email && u.email.toLowerCase().includes(search.toLowerCase()))
  );

  return (
    <div className="user-admin-container">
      <h2 className="user-admin-title">用户管理</h2>
      <div className="user-admin-bar">
        <input
          className="user-admin-search"
          type="text"
          placeholder="搜索用户名或邮箱"
          value={search}
          onChange={e => setSearch(e.target.value)}
        />
      </div>
      {loading ? <LoadingSpinner /> : (
        <div className="user-admin-list">
          <table className="user-admin-table">
            <thead>
              <tr>
                <th>ID</th>
                <th>用户名</th>
                <th>邮箱</th>
                <th>角色</th>
                <th>注册时间</th>
                <th>最后登录</th>
                <th>操作</th>
              </tr>
            </thead>
            <tbody>
              {filteredUsers.map(user => (
                <tr key={user.id} className={user.is_admin ? 'admin-row' : ''}>
                  <td>{user.id}</td>
                  <td>{user.username}</td>
                  <td>{user.email || '—'}</td>
                  <td>{user.is_admin ? '管理员' : '普通用户'}</td>
                  <td>{user.created_at ? new Date(user.created_at).toLocaleString() : '—'}</td>
                  <td>{user.last_login ? new Date(user.last_login).toLocaleString() : '—'}</td>
                  <td>
                    {user.is_admin ? (
                      <span className="admin-badge">超级管理员</span>
                    ) : (
                      <button className="admin-btn edit-btn" onClick={() => openEditModal(user)}>编辑</button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {editUserId && (
        <div className="edit-user-modal">
          <div className="edit-user-title">编辑用户</div>
          <label className="edit-label">邮箱：</label>
          <input
            className="edit-input"
            type="email"
            value={editEmail}
            onChange={e => setEditEmail(e.target.value)}
            placeholder="输入新邮箱"
          />
          <label className="edit-label">新密码：</label>
          <input
            className="edit-input"
            type="password"
            value={editPassword}
            onChange={e => setEditPassword(e.target.value)}
            placeholder="输入新密码（可选）"
          />
          <div className="edit-user-actions">
            <button className="admin-btn confirm-btn" onClick={handleEditSave}>保存</button>
            <button className="admin-btn delete-btn" onClick={handleEditDelete}>删除</button>
            <button className="admin-btn cancel-btn" onClick={closeEditModal}>取消</button>
          </div>
        </div>
      )}
    </div>
  );
}

export default UserAdminPage; 