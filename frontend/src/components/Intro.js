import React from 'react';
import './Intro.css';
import { useNavigate } from 'react-router-dom';

export default function Intro() {
  const navigate = useNavigate();
  return (
    <div className="intro-container">
      <h1>基于TimeXer的电力负荷预测系统</h1>
      
      <section className="intro-section">
        <h2>主要功能</h2>
        <ul>
          <li>
            <span className="feature-icon">📂</span>
            <div className="feature-content">
              <h3>数据集上传与管理</h3>
              <p>支持CSV格式数据集的上传、预览、删除与选择。快速查看数据集信息和预览内容。</p>
            </div>
          </li>
          <li>
            <span className="feature-icon">🔮</span>
            <div className="feature-content">
              <h3>电力负荷预测</h3>
              <p>基于深度学习模型TimeXer（时间序列预测模型）对所选数据集进行多时段（6/12/24小时）负荷预测。</p>
            </div>
          </li>
          <li>
            <span className="feature-icon">📈</span>
            <div className="feature-content">
              <h3>预测结果可视化</h3>
              <p>通过精美图表直观展示预测结果与实际值的对比，帮助理解模型效果。</p>
            </div>
          </li>
          <li>
            <span className="feature-icon">📋</span>
            <div className="feature-content">
              <h3>历史记录管理</h3>
              <p>自动保存每次预测结果，支持历史记录的浏览与删除，便于对比不同预测。</p>
            </div>
          </li>
          <li>
            <span className="feature-icon">🔒</span>
            <div className="feature-content">
              <h3>用户登录与权限管理</h3>
              <p>保障数据安全，支持多用户独立操作，个人中心方便管理您的账户信息。</p>
            </div>
          </li>
        </ul>
      </section>
      
      <section className="intro-section">
        <h2>使用流程</h2>
        <div className="process-flow">
          <div className="process-step">
            <div className="step-number">1</div>
            <div className="step-content">
              <h3>注册登录</h3>
              <p>创建您的账号并登录系统</p>
            </div>
          </div>
          <div className="process-arrow">→</div>
          <div className="process-step">
            <div className="step-number">2</div>
            <div className="step-content">
              <h3>上传数据</h3>
              <p>上传符合要求的CSV数据集</p>
            </div>
          </div>
          <div className="process-arrow">→</div>
          <div className="process-step">
            <div className="step-number">3</div>
            <div className="step-content">
              <h3>配置预测</h3>
              <p>选择数据集和预测参数</p>
            </div>
          </div>
          <div className="process-arrow">→</div>
          <div className="process-step">
            <div className="step-number">4</div>
            <div className="step-content">
              <h3>查看结果</h3>
              <p>分析预测结果并展示可视化图表</p>
            </div>
          </div>
        </div>
      </section>
      
      <section className="intro-section">
        <h2>界面说明</h2>
        <div className="interface-cards">
          <div className="interface-card" onClick={() => navigate('/')}
            tabIndex={0} role="button" aria-label="首页" >
            <div className="interface-icon">🏠</div>
            <h3>首页</h3>
            <p>数据集上传、选择与预测的主操作区</p>
          </div>
          <div className="interface-card" onClick={() => navigate('/history')}
            tabIndex={0} role="button" aria-label="历史记录" >
            <div className="interface-icon">📜</div>
            <h3>历史记录</h3>
            <p>查看和管理所有历史预测任务及结果</p>
          </div>
          <div className="interface-card" onClick={() => navigate('/profile')}
            tabIndex={0} role="button" aria-label="个人中心" >
            <div className="interface-icon">👤</div>
            <h3>个人中心</h3>
            <p>管理个人账户信息和统计数据</p>
          </div>
          <div className="interface-card" onClick={() => navigate('/comments')}
            tabIndex={0} role="button" aria-label="用户反馈" >
            <div className="interface-icon">💬</div>
            <h3>用户反馈</h3>
            <p>提交使用意见和建议</p>
          </div>
        </div>
      </section>
      
      <div className="intro-footer">
        <p>如有疑问请联系开发者微信：WSL1393720</p>
        <p>版本：1.0.0 · 更新日期：2025年05月</p>
      </div>
    </div>
  );
} 