# AIA-st3-midterm-face

![](https://i.imgur.com/i8R4NKN.jpg)

台灣人工智慧學校南部分校技術班第三期期中考  
<此repo留作紀錄用>  

> 題目:針對五位明星進行人臉辨識  
> 結果:  
> public-0.96183  
> private-0.88599  

作法:  
1. 單純直接進行CNN辨識，一般僅能達到6x~7x%準確率  
2. 進行特徵工程：使用dlib庫，並用人家預訓練好的CNN架構方式偵測人臉位置進行切割，手動篩選切割不佳者並自行補完沒切出的圖，生成新的train/test set(face detection那個notebook中有三種切法，可以自行嘗試)  
3. 利用transfer learning載入別人預訓練好的模型(VGG-face)進行feature extraction並連接自己的classfier  

以下是將dlib庫作CUDA compile的方式，若不用CUDA跑切幾百張圖就要幾小時  

`sudo apt-get update`  
`sudo apt-get install libblas-dev liblapack-dev libatlas-base-dev`  
`git clone https://github.com/davisking/dlib.git`  
`cd dlib`  
編輯 `setup.py`  在最上方(import os後)加 `os.environ["CC"] = "gcc-7"`  
`mkdir build && cd build`
```
cmake .. -DCUDA_HOST_COMPILER=/usr/bin/gcc-7 -DCMAKE_PREFIX_PATH=/usr/lib/x86_64-linux-gnu/ -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1 -DUSE_F16C=1
```  
`cmake --build . --config Release` #這時會靠 nvcc   
`sudo ldconfig`
`cd ..`
```
python setup.py install --record files.txt --compiler-flags "DCUDA_HOST_COMPILER=/usr/bin/gcc-7"
```