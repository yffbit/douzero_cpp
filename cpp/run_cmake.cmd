@echo off
set BUILD_TYPE=Release
set BUILD_DIR=build
set PREFIX_PATH=D:/ProgramFiles/Anaconda3/Lib/site-packages/torch/share/cmake
if "%1" == "Debug" (
    goto DEBUG
) else (
    goto BUILD
)

:DEBUG
set BUILD_TYPE=Debug
set BUILD_DIR=%BUILD_DIR%_debug
set PREFIX_PATH=D:/ProgramFiles/libtorch_debug/share/cmake
set PATH=%libtorch_debug%\bin;%libtorch_debug%\lib;%PATH%

:BUILD
if not exist %BUILD_DIR% mkdir %BUILD_DIR%
del /q %BUILD_DIR%\*
cd %BUILD_DIR%
cmake .. -DCMAKE_PREFIX_PATH="%PREFIX_PATH%" -DCMAKE_BUILD_TYPE="%BUILD_TYPE%" -G "Visual Studio 16 2019" -A x64
cd ..
