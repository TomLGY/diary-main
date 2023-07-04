## 1.导入.aar文件

将下载好的.aar文件放入app/libs下，如图

![image-20230704215314798](../../figs.assets/image-20230704215314798.png)

## 2.在app/build.gradle dependencies中加入：

implementation fileTree(dir: "libs", include: ["*.jar","*.aar"])

![image-20230704215616003](../../figs.assets/image-20230704215616003.png)

## 3.重新编译工程

编译后在External Libraries中可以找到

![image-20230704215806580](../../figs.assets/image-20230704215806580.png)
