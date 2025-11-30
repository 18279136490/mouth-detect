@echo off
echo 正在清理旧的构建文件...
rmdir /s /q build
rmdir /s /q dist
echo 正在打包程序...
pyinstaller --clean mouth_detection.spec
echo 打包完成！
pause