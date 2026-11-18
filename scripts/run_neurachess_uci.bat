@echo off
cd /d "%~dp0.."
set PYTHONPATH=%~dp0..\src
set PYTHONIOENCODING=utf-8
python -u -m neurachess.uci.engine_uci
