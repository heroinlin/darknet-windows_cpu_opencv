# Yolo-Windows
# "You Only Look Once: Unified, Real-Time Object Detection"
A yolo windows version (for object detection)

Contributtors: https://github.com/pjreddie/darknet/graphs/contributors

This repository is forked from: https://github.com/frischzenger/yolo-windows
Which is forked from Linux-version: https://github.com/pjreddie/darknet

##### Requires: 
* **MS Visual Studio 2013 (v120)**
* **CUDA 6.5 for Windows x64**: https://developer.nvidia.com/cuda-toolkit-65
* **OpenCV 2.4.9**: https://sourceforge.net/projects/opencvlibrary/files/opencv-win/2.4.9/opencv-2.4.9.exe/download
  - To compile without OpenCV - remove define OPENCV from: Visual Studio->Project->Properties->C/C++->Preprocessor
  - To compile with different OpenCV version - change in file yolo_kernels.cu each string **#pragma comment(lib, "opencv_core249.lib")** from 249 to required version.
  - With OpenCV will show image or video detection in window and store result to test_dnn_out.avi

##### Pre-trained models for different cfg-files can be downloaded from (smaller -> faster & lower quality):
* `yolo.cfg` (1 GB) - require 4 GB GPU-RAM: http://pjreddie.com/media/files/yolo.weights
* `yolo-small.cfg` (359 MB) - require 2 GB GPU-RAM: http://pjreddie.com/media/files/yolo-small.weights
* `yolo-tiny.cfg` (172 MB) - require 1 GB GPU-RAM: http://pjreddie.com/media/files/yolo-tiny.weights

##### Examples of results:

[![Everything Is AWESOME](http://img.youtube.com/vi/Gl1rxvEvgEs/0.jpg)](https://youtu.be/Gl1rxvEvgEs "Everything Is AWESOME")

Others: https://www.youtube.com/channel/UC7ev3hNVkx4DzZ3LO19oebg

More details: http://pjreddie.com/darknet/yolo/

##### Example of usage in cmd-files:
* `darknet.cmd` - initialization with 1 GB model yolo.weights & yolo.cfg and get command prompt: enter the jpg-image-filename
* `darknet_demo.cmd` - initialization with 1 GB model yolo.weights & yolo.cfg and play your video file which you must rename to: test.mp4
* `darknet_demo_small.cmd` - initialization with 359 MB small model yolo-small.weights & yolo-small.cfg and play your video file which you must rename to: test.mp4
* `darknet_net_cam.cmd` - initialization with 1 GB model, play video from network video-camera mjpeg-stream and store result to: test_dnn_out.avi

##### For using network video-camera mjpeg-stream with any Android smartphone:

1. Download for Android phone mjpeg-stream soft: IP Webcam / Smart WebCam


 Smart WebCam - preferably: https://play.google.com/store/apps/details?id=com.acontech.android.SmartWebCam
 IP Webcam: https://play.google.com/store/apps/details?id=com.pas.webcam

2. Connect your Android phone to computer by WiFi (through a WiFi-router) or USB
3. Start Smart WebCam on your phone
4. Replace the address below, on shown in the phone application (Smart WebCam) and launch:

```
darknet.exe yolo demo yolo.cfg yolo.weights http://192.168.0.80:8080/video?dummy=param.mjpg -i 0
```

Windows (Visual Studio) Support
Now yolo supports windows with Visual Studio. Just change the include directories / library directories to your own ones, 
and compile the codes with X64 Release mode. You may need pthread-windows while compiling and linking
