@echo off
gcc -O2 -o predict.exe predict.c -lm
if %errorlevel%==0 echo Built predict.exe
