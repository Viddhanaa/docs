@echo off
echo ========================================
echo   Logo Preview - VIDDHANA Documentation
echo ========================================
echo.
echo Starting documentation server with new logo...
echo.
cd /d K:\Viddhana_git\docs\viddhana\docs
echo Opening http://localhost:3030 in 10 seconds...
echo.
start /B npm start
timeout /t 10 /nobreak > nul
start http://localhost:3030
echo.
echo Documentation server is running!
echo Press Ctrl+C to stop the server
pause

