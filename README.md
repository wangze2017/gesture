# 基于opencv和paddlex的手势识别demo

### 文件/目录说明：

1. gethand.py  
	从视屏目录videos里截取手部图片，并保存在hands目录内。 
	
2. hand.py  
    封装了检测手部位置的方法，使用了开源的 [mediapipe](https://google.github.io/mediapipe/) 。
    
3. window.ui  
   使用 [PyQt5](https://pypi.org/project/PyQt5/) 生成的窗口UI文件。
   
4. window.py  
   使用PyQt5由UI文件生成的py文件。
   
5. main.py  
   程序的主入口，包含界面的逻辑代码。

6. models  
   由Paddlex训练出的模型。

7. test  
   用来测试项目及项目模型的图片。

### 视屏文件说明

**本版本只含两个测试视频目录，且模型文件识别的手势为：‘不’，‘好’，‘意’，‘思’**

每个手势有多个视屏文件，被包含在同一个目录内，目录命名为001-053，其命名数字代表了该手势的标签。  

|数字|手势|数字  |手势  |数字  |手势  |数字 |手势 |数字 |手势 |
|----| ---|---- | ---- | ----|----|----|---- |----| ----|
| 1  | 你 | 2 | 好 | 3 | 谢谢 |4 |再见 |5 |叫 |
| 6 | 什么 | 7 | 的 | 8 | 我 |9 |是 |10 |学生 |
| 11 | 不 | 12 | 意思 | 13 | 对不起 |14 |反对 |15 |打 |
| 16 | 电话 | 17 | 通知 | 18 | 同意 |19 |想法 |20 |很 |
| 21 | 高兴 | 22 | 认识 | 23 | 没 |24 |关系 |25 |可以 |
| 26 | 帮 | 27 | 教师 | 28 | 职业 |29 |什么 |30 |去 |
| 31 | 坐 | 32 | 飞机 | 33 | 开汽车 |34 |家 |35 |在 |
| 36 | 天津 | 37 | 多、几 | 38 | 大 |39 |时候 |40 |喜欢 |
| 41 | 游泳 | 42 | 朋友 | 43 | 现在、今天 |44 |星期 |45 |星期一 |
| 46 | 星期二 | 47 | 星期三 | 48 | 星期四 |49 |星期五 |50 |星期六 |
| 51 | 星期日 | 52 | 名字 |  | 联系 | | | | |









