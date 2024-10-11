const express = require('express');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware để phục vụ các tệp tĩnh từ thư mục 'public'
app.use(express.static(path.join(__dirname, 'public')));

// Route để phục vụ 'index.html'
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Khởi động máy chủ
app.listen(PORT, () => {
    console.log(`Server đang chạy tại http://localhost:${PORT}`);
});
