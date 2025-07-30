fetch('https://bee-touched-mink.ngrok-free.app/recommend', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      question: "我需要一件白色T恤"
    })
  })
  .then(response => response.json())
  .then(data => {
    console.log('返回结果:', data);
    
    // 检查响应状态
    if (data.code === 200) {
      console.log('✅ 请求成功');
      console.log('推荐内容:', data.data.answer);
      console.log('相关索引:', data.data.indexes);
    } else {
      console.log('❌ 请求失败');
      console.log('错误信息:', data.message);
    }
  })
  .catch(error => {
    console.error('请求出错:', error);
  });