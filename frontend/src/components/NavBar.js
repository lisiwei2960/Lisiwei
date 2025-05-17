import React from 'react';
import { Link } from 'react-router-dom';

export default function NavBar({ onLogout, isAdmin }) {
  return (
    <nav className="main-nav">
      <Link to="/" className="nav-link">首页</Link>
      <Link to="/history" className="nav-link">历史记录</Link>
      <Link to="/profile" className="nav-link">个人中心</Link>
      <Link to="/comments" className="nav-link">用户反馈</Link>
      {isAdmin && (
        <Link to="/admin/users" className="nav-link">用户管理</Link>
      )}
      <Link to="/intro" className="nav-link">功能介绍</Link>
      <button onClick={onLogout} className="logout-btn">退出登录</button>
    </nav>
  );
} 