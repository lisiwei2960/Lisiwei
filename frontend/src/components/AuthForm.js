import React from 'react';

export default function AuthForm({
  authTab,
  setAuthTab,
  username,
  setUsername,
  password,
  setPassword,
  handleLogin,
  handleRegister
}) {
  return (
    <div className="auth-page-flex">
      <div className="auth-intro-panel">
        <h2 className="auth-intro-title">电力负荷预测系统</h2>
        <div className="auth-intro-section">
          <h3>主要功能</h3>
          <ul>
            <li><b>数据集上传与管理：</b>支持CSV格式数据集的上传、预览、删除与选择。</li>
            <li><b>电力负荷预测：</b>基于深度学习模型TimeXer对所选数据集进行多时段（6/12/24小时）负荷预测。</li>
            <li><b>预测结果可视化：</b>展示预测曲线图表，便于直观理解模型效果。</li>
            <li><b>历史记录管理：</b>自动保存每次预测结果，支持历史记录的浏览与删除。</li>
            <li><b>用户登录与权限管理：</b>保障数据安全，支持多用户独立操作。</li>
          </ul>
        </div>
        <div className="auth-intro-section">
          <h3>使用流程</h3>
          <ol>
            <li>注册并登录账号</li>
            <li>上传符合要求的CSV数据集</li>
            <li>选择数据集，设置预测参数，发起预测</li>
            <li>查看预测结果的可视化图表</li>
            <li>在历史记录中管理和回溯预测任务</li>
            <li>管理员账号密码：admin/123456</li>
          </ol>
        </div>
      </div>
      <div className="auth-container" style={{display:'flex',flexDirection:'column',justifyContent:'center',alignItems:'center',minWidth:340,maxWidth:400,margin:'auto',boxShadow:'0 8px 32px rgba(24,144,255,0.10)'}}>
        <div className="auth-tabs">
          <button 
            className={authTab === 'login' ? 'active' : ''} 
            onClick={() => setAuthTab('login')}
            style={{minWidth:90,fontSize:18}}
          >
            登录
          </button>
          <button 
            className={authTab === 'register' ? 'active' : ''} 
            onClick={() => setAuthTab('register')}
            style={{minWidth:90,fontSize:18}}
          >
            注册
          </button>
        </div>
        <div className="auth-form" style={{background:'#fff',borderRadius:18,boxShadow:'0 4px 0px rgba(24,144,255,0.10)',padding:'36px 32px 28px 32px',width:'100%',maxWidth:340,display:'flex',flexDirection:'column',alignItems:'center',gap:24}}>
          <h2 style={{margin:'0 0 -15px 0',fontWeight:700,color:'#1890ff',fontSize:24,letterSpacing:1}}>欢迎{authTab==='login'?'登录':'注册'}</h2>
          <p style={{margin:'-5px 0 -15px 0',color:'#888',fontSize:14}}>{authTab==='login'?'请输入账号和密码登录':'请填写信息完成注册'}</p>
          <input
            type="text"
            placeholder="用户名"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            style={{marginBottom:-10,fontSize:16,height:44}}
          />
          <input
            type="password"
            placeholder="密码"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            style={{marginBottom:-10,fontSize:16,height:44}}
          />
          <button onClick={authTab === 'login' ? handleLogin : handleRegister} style={{fontSize:18,height:44,lineHeight:'44px',marginTop:8,textAlign:'center',padding:0}}>
            {authTab === 'login' ? '登录' : '注册'}
          </button>
        </div>
      </div>
    </div>
  );
} 