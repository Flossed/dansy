@echo off
echo === Mini LLM TensorFlow.js Application ===
echo.

REM Check if node_modules exists
if not exist node_modules (
    echo Installing dependencies...
    npm install
    echo.
)

echo Starting server...
echo.
echo Once started, open your browser to: http://localhost:3000
echo.
echo Press Ctrl+C to stop the server
echo.

npm start
