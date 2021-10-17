# [DouZero](https://github.com/kwai/DouZero)的cpp版本
多线程，支持GPU训练

## 编译环境
* cmake
* Visual Studio 2019
* PyTorch或者LibTorch

目前只在Windows上测试过，需要安装[PyTorch](https://pytorch.org)或者LibTorch。PyTorch自带Release版本，LibTorch有[Release](https://download.pytorch.org/libtorch/cu102/libtorch-win-shared-with-deps-1.9.1%2Bcu102.zip)和[Debug](https://download.pytorch.org/libtorch/cu102/libtorch-win-shared-with-deps-debug-1.9.1%2Bcu102.zip)两个版本。如果需要调试，则需要安装LibTorch的Debug版本。不能用Release版本进行调试，可能会出现一些奇怪的bug。建议将不同版本的路径都加入环境变量，例如新建环境变量`libtorch`和`libtorch_debug`分别指向Release版本和Debug版本的路径，切换的时候就比较方便。cmd里面切换版本
```
set PATH=%libtorch%\bin;%libtorch%\lib;%PATH%
```
或者
```
set PATH=%libtorch_debug%\bin;%libtorch_debug%\lib;%PATH%
```
对于PyTorch，如果是通过anaconda安装的，路径应该在`Anaconda根目录\Lib\site-packages\torch\`，例如
```
D:\ProgramFiles\Anaconda3\Lib\site-packages\torch\
```
执行`run_cmake.cmd`后创建Visual Studio项目。如果需要调试，则执行`run_cmake.cmd Debug`。Visual Studio里面调试时需要将Debug版本加入环境，项目属性->调试->环境，加入以下代码
```
PATH=%libtorch_debug%\bin;%libtorch_debug%\lib;%PATH%
```

## 出牌规则
[botzone平台上的FightTheLandlord](https://wiki.botzone.org.cn/index.php?title=FightTheLandlord)

## 训练
参数基本上和DouZero一致，通过文本文件传入。
```
dou_train.exe config/config.txt
```
## 与baseline模型比较
baseline模型是Pytorch生成的，需要转化为[Torch Script Model](https://pytorch.org/tutorials/advanced/cpp_export.html)之后，才能被LibTorch导入，示例代码见`tool.py`。通过`tool.py`得到转化后的模型文件，将其路径写入`evaluate_config.txt`并且设置`jit=true`，再执行
```
evaluate_two_agent.exe config/evaluate_config.txt
```
对抗时，两个农民为同一个agent。同样的手牌，两个agent会交换身份进行对抗。也就是每次对抗由两局游戏组成，第二局的牌和第一局一样，只是两个agent的身份交换。