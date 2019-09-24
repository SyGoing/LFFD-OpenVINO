[![LICENSE](https://img.shields.io/badge/license-NPL%20(The%20996%20Prohibited%20License)-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)

## the OpenVINO implemententation of LFFD  
  paper:[LFFD: A Light and Fast Face Detector for Edge Devices](https://arxiv.org/abs/1904.10633)
  
  official github: [LFFD](https://github.com/YonghaoHe/A-Light-and-Fast-Face-Detector-for-Edge-Devices)
  
  ncnn implementation is [here](https://github.com/SyGoing/LFFD-with-ncnn)
  
  MNN implementation is [here](https://github.com/SyGoing/LFFD-MNN)
  
## Prerequirements
  Please refer the official OPenVINO‘s [DOC](https://software.intel.com/en-us/openvino-toolkit/documentation/get-started) to install openvino.
  In that documentation, you will find how to convert the official mxnet model to openvino. 
  And,before you convert the mxnet model ,you need to modify the symbol.json as follows:
  
  * First ,follow the author's original  github to build the devolopment environment.
  * Modify symbol_10_320_20L_5scales_v2.py (your_path/A-Light-and-Fast-Face-Detector-for-Edge-Devices\face_detection\symbol_farm) 
      in function loss_branch,Note out（注释掉） the line 57(predict_score = mxnet.symbol.slice_axis(predict_score, axis=1, begin=0, end=1)
	  in function get_net_symbol, Note out（注释掉）the line 99(data = (data - 127.5) / 127.5,preprocess).
  * Next,in this path , by doing "python symbol_10_320_20L_5scales_v2.py	",generate the symbol.json. symbol_10_560_25L_8scales_v1.py do the same thing .

 ## Inference time on CPU i7 7700 
  The time is average time. 
 set the mode CPU. When setting it GPU, it uses the intel graphic gpu

* v2:

Resolution->|320×240|640×480|1280x720|1920x1080
------------|------------|-----------|-----------|------------
 **LFFD**|11.20ms(89.28 FPS)|44.61ms(22.41 FPS)|128.61ms(7.78 FPS)|288.01ms(3.47 FPS)

* v1:

  Resolution->|320×240|640×480|1280x720|1920x1080
------------|-------|--------|---------|------------------
 **LFFD**|15.25ms(65.57 FPS)|60.19ms(16.61 FPS)|179.32ms(5.58 FPS)|409.01ms(2.44 FPS)
