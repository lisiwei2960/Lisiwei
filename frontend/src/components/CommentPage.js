import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './CommentPage.css';
import { CSSTransition } from 'react-transition-group';

export default function CommentPage({ showMessage }) {
  const [comments, setComments] = useState([]);
  const [content, setContent] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [currentUser, setCurrentUser] = useState('');
  const [isAdmin, setIsAdmin] = useState(false);
  const [confirmDeleteId, setConfirmDeleteId] = useState(null);
  const [replyingId, setReplyingId] = useState(null);
  const [replyContent, setReplyContent] = useState('');

  useEffect(() => {
    const username = localStorage.getItem('username');
    setCurrentUser(username || '');
    setIsAdmin(localStorage.getItem('is_admin') === 'true');
  }, []);

  // 获取所有评论（嵌套）
  const fetchComments = async () => {
    setLoading(true);
    try {
      const res = await axios.get('http://localhost:5000/comments');
      // 反转一级评论列表，但保留各评论内部的回复顺序不变
      const sortedComments = [...res.data].reverse();
      setComments(sortedComments);
    } catch (err) {
      setError('获取评论失败');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchComments();
  }, []);

  // 新增：自适应高度的textarea处理函数
  function autoResizeTextarea(e) {
    const textarea = e.target;
    textarea.style.height = 'auto';
    textarea.style.height = textarea.scrollHeight + 'px';
  }

  // 发表评论
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!content.trim()) {
      if (showMessage) showMessage('反馈内容不能为空', 'error');
      return;
    }
    setError('');
    try {
      const token = localStorage.getItem('token');
      await axios.post('http://localhost:5000/comments', { content }, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setContent('');
      if (showMessage) showMessage('评论成功', 'success');
      fetchComments();
    } catch (err) {
      setError(err.response?.data?.error || '评论失败');
    }
  };

  // 删除评论
  const handleDelete = async (id) => {
    setConfirmDeleteId(id);
  };
  const confirmDelete = async () => {
    if (!confirmDeleteId) return;
    try {
      const token = localStorage.getItem('token');
      await axios.delete(`http://localhost:5000/comments/${confirmDeleteId}`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      if (showMessage) showMessage('评论已删除', 'success');
      setConfirmDeleteId(null);
      fetchComments();
    } catch (err) {
      setError(err.response?.data?.error || '删除失败');
      setConfirmDeleteId(null);
    }
  };

  // 回复评论
  const handleReply = async (parentId) => {
    if (!replyContent.trim()) {
      if (showMessage) showMessage('回复内容不能为空', 'error');
      return;
    }
    try {
      const token = localStorage.getItem('token');
      await axios.post('http://localhost:5000/comments/reply', { content: replyContent, parent_id: parentId }, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setReplyContent('');
      setReplyingId(null);
      if (showMessage) showMessage('回复成功', 'success');
      fetchComments();
    } catch (err) {
      setError(err.response?.data?.error || '回复失败');
    }
  };

  // 递归渲染评论树
  const renderComments = (commentList) => (
    commentList.map(c => (
      <div key={c.id} className="comment-card animate-fade-in">
        <div className="comment-header">
          <span className="comment-username">{c.username}</span>
          <div className="comment-actions">
            {(currentUser && (c.username === currentUser || isAdmin)) && (
              <button onClick={() => setConfirmDeleteId(c.id)} className="delete-btn ripple">删除</button>
            )}
            {(!!currentUser) && (
              <button onClick={() => setReplyingId(c.id)} className="delete-btn reply ripple">回复</button>
            )}
          </div>
        </div>
        <div className="comment-content">{c.content}</div>
        <div className="comment-time">{(() => { const beijingTime = new Date(new Date(c.created_at).getTime() + 8 * 60 * 60 * 1000); return beijingTime.toLocaleString('zh-CN', { hour12: false }); })()}</div>
        {replyingId === c.id && (
          <div className="reply-box">
            <textarea
              className="comment-input"
              value={replyContent}
              onChange={e => { setReplyContent(e.target.value); autoResizeTextarea(e); }}
              placeholder="请输入回复内容..."
              rows={2}
              style={{marginTop: 8, overflow: 'hidden'}}
            />
            <button className="btn-apple comment-btn" style={{marginTop: 6}} onClick={() => handleReply(c.id)}>提交回复</button>
            <button className="btn-apple comment-btn" style={{marginTop: 6, marginLeft: 8, background: '#eee', color: '#666'}} onClick={() => {setReplyingId(null);setReplyContent('');}}>取消</button>
          </div>
        )}
        {c.children && c.children.length > 0 && (
          <div className="comment-children">
            {renderComments(c.children)}
          </div>
        )}
      </div>
    ))
  );

  return (
    <div className="comment-page-container">
      <h2 className="comment-title">用户反馈</h2>
      <form onSubmit={handleSubmit} className="comment-form">
        <textarea
          value={content}
          onChange={e => { setContent(e.target.value); autoResizeTextarea(e); }}
          rows={4}
          className="comment-input"
          placeholder="请输入您的反馈..."
          style={{overflow: 'hidden'}}
        />
        <button type="submit" className="btn-apple comment-btn">发表反馈</button>
      </form>
      {error && <div className="comment-message error-message">{error}</div>}
      <div className="comment-list">
        {loading ? <div className="comment-loading">加载中...</div> : (
          comments.length === 0 ? <div className="comment-empty">暂无评论</div> : (
            renderComments(comments)
          )
        )}
      </div>
      <CSSTransition
        in={!!confirmDeleteId}
        timeout={350}
        classNames="apple-modal"
        unmountOnExit
      >
        <div className="confirm-dialog apple-modal">
          <p>确定要删除这条评论吗？</p>
          <div className="confirm-actions">
            <button onClick={confirmDelete}>确定</button>
            <button onClick={() => setConfirmDeleteId(null)}>取消</button>
          </div>
        </div>
      </CSSTransition>
    </div>
  );
} 