@ECHO OFF
@SET PYTHONIOENCODING=utf-8
@SET PYTHONUTF8=1
@FOR /F "tokens=2 delims=:." %%A in ('chcp') do for %%B in (%%A) do set "_CONDA_OLD_CHCP=%%B"
@chcp 65001 > NUL
@CALL "D:\anaconda3\condabin\conda.bat" activate "e:\学习\基于大模型的电力负荷预测方法研究与开发\时间序列集合\.conda"
@IF %ERRORLEVEL% NEQ 0 EXIT /b %ERRORLEVEL%
@e:\学习\基于大模型的电力负荷预测方法研究与开发\时间序列集合\.conda\python.exe -Wi -m compileall -q -l -i C:\Users\29600\AppData\Local\Temp\tmp72gdpn09 -j 0
@IF %ERRORLEVEL% NEQ 0 EXIT /b %ERRORLEVEL%
@chcp %_CONDA_OLD_CHCP%>NUL
