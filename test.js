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
  });