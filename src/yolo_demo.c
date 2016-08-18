#include "network.h"
#include "detection_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
//#include <sys/time.h>
#include <stdio.h>
#include <time.h>
#include <winsock.h>


//#pragma comment(lib, "opencv_core249.lib")  
//#pragma comment(lib, "opencv_imgproc249.lib")  
//#pragma comment(lib, "opencv_objdetect249.lib")  
//#pragma comment(lib, "opencv_gpu249.lib")  
//#pragma comment(lib, "opencv_features2d249.lib")  
//#pragma comment(lib, "opencv_highgui249.lib")  
////#pragma comment(lib, "opencv_ml249.lib")  
//#pragma comment(lib, "opencv_stitching249.lib")  
//#pragma comment(lib, "opencv_nonfree249.lib")  
////#pragma comment(lib, "opencv_superres249.lib")  
//#pragma comment(lib, "opencv_calib3d249.lib")  
//#pragma comment(lib, "opencv_flann249.lib")  
////#pragma comment(lib, "opencv_contrib249.lib")  
////#pragma comment(lib, "opencv_legacy249.lib")  
//#pragma comment(lib, "opencv_photo249.lib")  
//#pragma comment(lib, "opencv_video249.lib")  

#ifdef OPENCV

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
image ipl_to_image(IplImage* src);
void convert_yolo_detections(float *predictions, int classes, int num, int square, int side, int w, int h, float thresh, float **probs, box *boxes, int only_objectness);
void draw_yolo(image im, int num, float thresh, box *boxes, float **probs);
void draw_detections_net(network *net, image im, int num, float thresh, box *boxes, float **probs, char **names, image *labels, int classes);
extern char *voc_names[];
extern image voc_labels[];
static float **probs;
static box *boxes;
static network net;
static network net1;
//static image in   ;
//static image in_r;
//static image in_s ;
//static image det  ;
//static image det_s;
//static image disp ;
//static CvCapture * cap;
static float fps = 0;
static float demo_thresh = 0;

//void *fetch_in_thread(void *ptr)
//{
//    in = get_image_from_stream(cap);
//	in_r = crop_image(in, 126,53,448,403);
//	in_s = resize_image(in_r, net.w, net.h);
//    //in_s = resize_image(in, net.w, net.h);
//    return 0;
//}
//
//void *detect_in_thread(void *ptr)
//{
//    float nms = .4;
//
//    detection_layer l = net.layers[net.n-1];
//    float *X = det_s.data;
//    float *predictions = network_predict(net, X);
//    free_image(det_s);
//    convert_yolo_detections(predictions, l.classes, l.n, l.sqrt, l.side, 1, 1, demo_thresh, probs, boxes, 0);
//    if (nms > 0) do_nms(boxes, probs, l.side*l.side*l.n, l.classes, nms);
//    printf("\033[2J");
//    printf("\033[1;1H");
//    printf("\nFPS:%.0f\n",fps);
//    printf("Objects:\n\n");
//
//
//    draw_detections(det, l.side*l.side*l.n, demo_thresh, boxes, probs, voc_names, voc_labels, 1);
//    return 0;
//}
//void *detect_sort_in_thread(void *ptr)
//{
//	float nms = .4;
//	detection_layer l = net.layers[net.n - 1];
//	float *X = det_s.data;
//	float *predictions = network_predict(net, X);
//	free_image(det_s);
//	convert_yolo_detections(predictions, l.classes, l.n, l.sqrt, l.side, 1, 1, demo_thresh, probs, boxes, 0);
//	//printf("demo_thresh=%f\n", demo_thresh);
//	if (nms > 0) do_nms(boxes, probs, l.side*l.side*l.n, l.classes, nms);
//	printf("\033[2J");
//	printf("\033[1;1H");
//	printf("\nFPS:%.0f\n", fps);
//	printf("Objects:\n\n");
//	draw_detections_net(&net1,det, l.side*l.side*l.n, demo_thresh, boxes, probs, voc_names, voc_labels, 1);
//	return 0;
//}

//void demo_yolo(char *cfgfile, char *weightfile, float thresh, int cam_index, char *filename)
//{
//    demo_thresh = thresh;
//    printf("YOLO demo\n");
//    net = parse_network_cfg(cfgfile);
//    if(weightfile){
//        load_weights(&net, weightfile);
//    }
//    set_batch_network(&net, 1);
//
//    srand(2222222);
//	int numFrames = 0;
//	char buff[256];
//	char *basename = buff;
//    if(filename){
//        cap = cvCaptureFromFile(filename);
//		basename = get_basename(filename);
//    }else{
//        cap = cvCaptureFromCAM(cam_index);
//		basename = "web_cam_YOLO";
//    }
//
//    if(!cap) error("Couldn't connect to webcam.\n");
//	basename = strcat(basename, "_result.avi");
//	cvNamedWindow(basename, CV_WINDOW_NORMAL);
//	cvResizeWindow(basename, 512, 512);
//	numFrames = (int)cvGetCaptureProperty(cap, CV_CAP_PROP_FRAME_COUNT);
//    detection_layer l = net.layers[net.n-1];
//    int j;
//
//    boxes = (box *)calloc(l.side*l.side*l.n, sizeof(box));
//    probs = (float **)calloc(l.side*l.side*l.n, sizeof(float *));
//    for(j = 0; j < l.side*l.side*l.n; ++j) probs[j] = (float *)calloc(l.classes, sizeof(float *));
//
//    pthread_t fetch_thread;
//    pthread_t detect_thread;
//
//    fetch_in_thread(0);
//    det = in;
//    det_s = in_s;
//
//
//    fetch_in_thread(0);
//    detect_in_thread(0);
//    disp = det;
//    det = in;
//    det_s = in_s;
//	int countFrames = 2;
//	while (countFrames<numFrames){
//		countFrames++;
//		//printf("countFrames=%d,numFrame=%d\n",countFrames,numFrames);//debuging
//        typedef struct timeval timeval;
//        timeval tval_before, tval_after, tval_result;
//        gettimeofday(&tval_before, NULL);
//        if(pthread_create(&fetch_thread, 0, fetch_in_thread, 0)) error("Thread creation failed");
//        if(pthread_create(&detect_thread, 0, detect_in_thread, 0)) error("Thread creation failed");
//        show_image(disp, basename);
//        free_image(disp);
//        cvWaitKey(1);
//        pthread_join(fetch_thread, 0);
//        pthread_join(detect_thread, 0);
//
//        disp  = det;
//        det   = in;
//        det_s = in_s;
//
//        gettimeofday(&tval_after, NULL);
//        timersub(&tval_after, &tval_before, &tval_result);
//        float curr = 1000000.f/((long int)tval_result.tv_usec);
//        fps = .9*fps + .1*curr;
//    }
//}

//void demo_yolo_fold(char *cfgfile, char *weightfile, float thresh, int cam_index, char *filename)
//{
//	demo_thresh = thresh;
//	printf("YOLO demo\n");
//	net = parse_network_cfg(cfgfile);
//	if (weightfile){
//		load_weights(&net, weightfile);
//	}
//	set_batch_network(&net, 1);
//	list *plist = get_paths(filename);
//	char **paths = (char **)list_to_array(plist);
//	int m = plist->size;
//	int i = 0;
//	srand(2222222);
//	char buff[256];
//	char *input = buff;
//	int numFrames = 0;
//	for (i = 0; i < m; i++)
//	{
//		if (paths[i])
//		{
//			strncpy(input, paths[i], 256);
//		}
//		cap = cvCaptureFromFile(input);
//
//		
//		if (!cap) error("Couldn't connect to webcam.\n");
//		char *basename = get_basename(input);
//		basename = strcat(basename,"_result.avi");
//		cvNamedWindow(basename, CV_WINDOW_NORMAL);
//		cvResizeWindow(basename, 512, 512);
//
//		numFrames = (int)cvGetCaptureProperty(cap, CV_CAP_PROP_FRAME_COUNT);
//		detection_layer l = net.layers[net.n - 1];
//		int j;
//
//		boxes = (box *)calloc(l.side*l.side*l.n, sizeof(box));
//		probs = (float **)calloc(l.side*l.side*l.n, sizeof(float *));
//		for (j = 0; j < l.side*l.side*l.n; ++j) probs[j] = (float *)calloc(l.classes, sizeof(float *));
//
//		pthread_t fetch_thread;
//		pthread_t detect_thread;
//
//		fetch_in_thread(0);
//		det = in;
//		det_s = in_s;
//
//
//		fetch_in_thread(0);
//		detect_in_thread(0);
//		disp = det;
//		det = in;
//		det_s = in_s;
//		int countFrames = 2;
//		while (countFrames < numFrames){
//			countFrames++;
//			printf("countFrames=%d,numFrame=%d\n",countFrames,numFrames);//debuging
//			typedef struct timeval timeval;
//			timeval tval_before, tval_after, tval_result;
//			gettimeofday(&tval_before, NULL);
//			if (pthread_create(&fetch_thread, 0, fetch_in_thread, 0)) error("Thread creation failed");
//			if (pthread_create(&detect_thread, 0, detect_in_thread, 0)) error("Thread creation failed");
//			show_image(disp, basename);
//			free_image(disp);
//			cvWaitKey(1);
//			pthread_join(fetch_thread, 0);
//			pthread_join(detect_thread, 0);
//
//			disp = det;
//			det = in;
//			det_s = in_s;
//
//			gettimeofday(&tval_after, NULL);
//			timersub(&tval_after, &tval_before, &tval_result);
//			float curr = 1000000.f / ((long int)tval_result.tv_usec);
//			fps = .9*fps + .1*curr;
//		}
//		cvDestroyWindow(basename);
//	}
//}
//void demo_yolo_fold1(char *cfgfile, char *weightfile, float thresh, int cam_index, char *filename)
//{
//	demo_thresh = thresh;
//	printf("YOLO demo\n");
//	net = parse_network_cfg(cfgfile);
//	if (weightfile){
//		load_weights(&net, weightfile);
//	}
//	set_batch_network(&net, 1);
//	list *plist = get_paths(filename);
//	char **paths = (char **)list_to_array(plist);
//	int m = plist->size;
//	int i = 0;
//	srand(2222222);
//	char buff[256];
//	char *input = buff;
//	int numFrames = 0;
//	for (i = 0; i < m; i++)
//	{
//		if (paths[i])
//		{
//			strncpy(input, paths[i], 256);
//		}
//
//		cap = cvCreateFileCapture(input);
//		//cap = cvCaptureFromFile(input);
//		if (!cap) error("Couldn't connect to webcam.\n");
//		char *basename = get_basename(input);
//		basename = strcat(basename, "_result.avi");
//		cvNamedWindow(basename, CV_WINDOW_NORMAL);
//		cvResizeWindow(basename, 512, 512);
//
//		numFrames = (int)cvGetCaptureProperty(cap, CV_CAP_PROP_FRAME_COUNT);
//		detection_layer l = net.layers[net.n - 1];
//		int j;
//
//		boxes = (box *)calloc(l.side*l.side*l.n, sizeof(box));
//		probs = (float **)calloc(l.side*l.side*l.n, sizeof(float *));
//		for (j = 0; j < l.side*l.side*l.n; ++j) probs[j] = (float *)calloc(l.classes, sizeof(float *));
//
//		pthread_t fetch_thread;
//		pthread_t detect_thread;
//
//		fetch_in_thread(0);
//		det = in_r;
//		det_s = in_s;
//	
//		fetch_in_thread(0);
//		detect_in_thread(0);
//		disp = det;
//		det = in_r;
//		det_s = in_s;
//		int countFrames = 0;
//		while (countFrames < numFrames-2){
//		    countFrames++;
//			printf("countFrames=%d,numFrame=%d\n", countFrames, numFrames);//debuging
//			typedef struct timeval timeval;
//			timeval tval_before, tval_after, tval_result;
//			gettimeofday(&tval_before, NULL);
//			if (pthread_create(&fetch_thread, 0, fetch_in_thread, 0)) error("Thread creation failed");
//			if (pthread_create(&detect_thread, 0, detect_in_thread, 0)) error("Thread creation failed");
//			show_image(disp, basename);
//			free_image(disp);
//			
//			
//			char savename[256];
//			sprintf(savename, "./results/%d_%07d.jpg",i, countFrames);
//			int key = cvWaitKey(0);
//			switch (key)
//			{
//			case 'd':{ countFrames -= 2; cvSetCaptureProperty(cap, CV_CAP_PROP_POS_FRAMES, countFrames); break; }
//			case 'D':{ countFrames -= 2; cvSetCaptureProperty(cap, CV_CAP_PROP_POS_FRAMES, countFrames); break; }
//			case 's':{ save_image_jpg(in, savename); break; }
//			case 'S':{ save_image_jpg(in, savename); break; }
//			default:{}
//			}
//			//cvWaitKey(1);
//			pthread_join(fetch_thread, 0);
//			pthread_join(detect_thread, 0);
//
//			disp = det;
//			det = in_r;
//			det_s = in_s;
//			gettimeofday(&tval_after, NULL);
//			timersub(&tval_after, &tval_before, &tval_result);
//			float curr = 1000000.f / ((long int)tval_result.tv_usec);
//			fps = .9*fps + .1*curr;
//		
//		}
//		free_image(det);
//		free_image(det_s);
//		cvReleaseCapture(&cap);
//		cvDestroyWindow(basename);
//	}
//}
//void demo_sort_fold(char *cfgfile, char *weightfile, char *filename, char *cfgfile1, char *weightfile1, float thresh, int cam_index)
//{
//	demo_thresh = thresh;
//	printf("YOLO demo\n");
//	net = parse_network_cfg(cfgfile);
//	if (weightfile){
//		load_weights(&net, weightfile);
//	}
//	set_batch_network(&net, 1);
//	net1= parse_network_cfg(cfgfile1);
//	if (weightfile1){
//		load_weights(&net1, weightfile1);
//	}
//	set_batch_network(&net1, 1);
//	list *plist = get_paths(filename);
//	char **paths = (char **)list_to_array(plist);
//	int m = plist->size;
//	int i = 0;
//	srand(2222222);
//	char buff[256];
//	char *input = buff;
//	int numFrames = 0;
//	for (i = 0; i < m; i++)
//	{
//		if (paths[i])
//		{
//			strncpy(input, paths[i], 256);
//		}
//		cap = cvCaptureFromFile(input);
//
//
//		if (!cap) error("Couldn't connect to webcam.\n");
//		char *basename = get_basename(input);
//		basename = strcat(basename, "_result.avi");
//		cvNamedWindow(basename, CV_WINDOW_NORMAL);
//		cvResizeWindow(basename, 512, 512);
//
//		numFrames = (int)cvGetCaptureProperty(cap, CV_CAP_PROP_FRAME_COUNT);
//		detection_layer l = net.layers[net.n - 1];
//		int j;
//
//		boxes = (box *)calloc(l.side*l.side*l.n, sizeof(box));
//		probs = (float **)calloc(l.side*l.side*l.n, sizeof(float *));
//		for (j = 0; j < l.side*l.side*l.n; ++j) probs[j] = (float *)calloc(l.classes, sizeof(float *));
//
//		pthread_t fetch_thread;
//		pthread_t detect_sort_thread;
//
//		fetch_in_thread(0);
//		det = in;
//		det_s = in_s;
//
//
//		fetch_in_thread(0);
//		detect_sort_in_thread(0);
//		disp = det;
//		det = in;
//		det_s = in_s;
//		int countFrames = 2;
//		while (countFrames < numFrames){
//			countFrames++;
//			printf("countFrames=%d,numFrame=%d\n", countFrames, numFrames);//debuging
//			typedef struct timeval timeval;
//			timeval tval_before, tval_after, tval_result;
//			gettimeofday(&tval_before, NULL);
//			if (pthread_create(&fetch_thread, 0, fetch_in_thread, 0)) error("Thread creation failed");
//			if (pthread_create(&detect_sort_thread, 0, detect_sort_in_thread, 0)) error("Thread creation failed");
//			show_image(disp, basename);
//			free_image(disp);
//			cvWaitKey(1);
//			pthread_join(fetch_thread, 0);
//			pthread_join(detect_sort_thread,0);
//
//			disp = det;
//			det = in;
//			det_s = in_s;
//
//			gettimeofday(&tval_after, NULL);
//			timersub(&tval_after, &tval_before, &tval_result);
//			float curr = 1000000.f / ((long int)tval_result.tv_usec);
//			fps = .9*fps + .1*curr;
//		}
//		cvDestroyWindow(basename);
//	}
//}

void demo_yolo(char *cfgfile, char *weightfile, float thresh, int cam_index, char *filename)
{
	demo_thresh = thresh;
	printf("YOLO demo\n");
	net = parse_network_cfg(cfgfile);
	if (weightfile){
		load_weights(&net, weightfile);
	}
	set_batch_network(&net, 1);
	srand(2222222);
	int numFrames = 0;
	float nms = .4;
	CvCapture  *cap1;
	char buff[256];
	char *basename = buff;
	if (filename){
		cap1 = cvCaptureFromFile(filename);
		basename = get_basename(filename);
	}
	else{
		cap1 = cvCaptureFromCAM(cam_index);
		basename = "web_cam_YOLO";
	}
	if (!cap1) error("Couldn't connect to webcam.\n");
	basename = strcat(basename, "_result.avi");
	cvNamedWindow(basename, CV_WINDOW_NORMAL);
	cvResizeWindow(basename, 512, 512);
	numFrames = (int)cvGetCaptureProperty(cap1, CV_CAP_PROP_FRAME_COUNT);
	detection_layer l = net.layers[net.n - 1];
	int j;

	boxes = (box *)calloc(l.side*l.side*l.n, sizeof(box));
	probs = (float **)calloc(l.side*l.side*l.n, sizeof(float *));
	for (j = 0; j < l.side*l.side*l.n; ++j) probs[j] = (float *)calloc(l.classes, sizeof(float *));
	int countFrames = 0;
	IplImage *frame = NULL;
	while (countFrames < numFrames){
		frame = cvQueryFrame(cap1);
		image im = ipl_to_image(frame);
		image im_s = resize_image(im, net.w, net.h);
		countFrames++;

		float *X = im_s.data;
		float *predictions = network_predict(net, X);
		convert_yolo_detections(predictions, l.classes, l.n, l.sqrt, l.side, 1, 1, demo_thresh, probs, boxes, 0);
		if (nms > 0) do_nms(boxes, probs, l.side*l.side*l.n, l.classes, nms);
		draw_detections(im, l.side*l.side*l.n, demo_thresh, boxes, probs, voc_names, voc_labels, 1);
		printf("countFrames=%d,numFrame=%d\n", countFrames, numFrames);//debuging
		show_image(im, basename);
		cvWaitKey(1);
		free_image(im_s);
		free_image(im);
	}
	cvReleaseCapture(&cap1);

}
void demo_yolo_fold(char *cfgfile, char *weightfile, float thresh, int cam_index, char *filename)
{
	demo_thresh = thresh;
	printf("YOLO demo\n");
	net = parse_network_cfg(cfgfile);
	if (weightfile){
		load_weights(&net, weightfile);
	}
	set_batch_network(&net, 1);
	list *plist = get_paths(filename);
	char **paths = (char **)list_to_array(plist);
	int m = plist->size;
	int i = 0;
	srand(2222222);
	char buff[256];
	char *input = buff;
	int numFrames = 0;
	float nms = .4;
	for (i = 0; i < m; i++)
	{
		if (paths[i])
		{
			strncpy(input, paths[i], 256);
		}

		CvCapture  *cap1 = cvCreateFileCapture(input);
		//cap = cvCaptureFromFile(input);
		if (!cap1) error("Couldn't connect to webcam.\n");

		numFrames = (int)cvGetCaptureProperty(cap1, CV_CAP_PROP_FRAME_COUNT);
		detection_layer l = net.layers[net.n - 1];
		int j;

		boxes = (box *)calloc(l.side*l.side*l.n, sizeof(box));
		probs = (float **)calloc(l.side*l.side*l.n, sizeof(float *));
		for (j = 0; j < l.side*l.side*l.n; ++j) probs[j] = (float *)calloc(l.classes, sizeof(float *));
		int countFrames = 0;
		IplImage *frame = NULL;
		while (countFrames < numFrames){
			frame = cvQueryFrame(cap1);
			image im = ipl_to_image(frame);
			image im_s = resize_image(im, net.w, net.h);
			countFrames++;

			float *X = im_s.data;
			float *predictions = network_predict(net, X);
			convert_yolo_detections(predictions, l.classes, l.n, l.sqrt, l.side, 1, 1, demo_thresh, probs, boxes, 0);
			if (nms > 0) do_nms(boxes, probs, l.side*l.side*l.n, l.classes, nms);
			draw_detections(im, l.side*l.side*l.n, demo_thresh, boxes, probs, voc_names, voc_labels, 1);
			printf("countFrames=%d,numFrame=%d\n", countFrames, numFrames);//debuging
			show_image(im, "YOLO");
			char savename[256];
			sprintf(savename, "./results/%d_%07d", i, countFrames);
			cvWaitKey(1);
			free_image(im_s);
			free_image(im);
		}
		cvReleaseCapture(&cap1);
	}
}
void demo_yolo_fold2(char *cfgfile, char *weightfile, float thresh, int cam_index, char *filename, int x, int y, int w, int h)
{
	demo_thresh = thresh;
	char Folder_name[256];
	printf("Please Enter the Folder to save pictures:\n");
	scanf("%s/n",&Folder_name);
	printf("YOLO demo\n");
	net = parse_network_cfg(cfgfile);
	if (weightfile){
		load_weights(&net, weightfile);
	}
	set_batch_network(&net, 1);
	list *plist = get_paths(filename);
	char **paths = (char **)list_to_array(plist);
	int m = plist->size;
	int i = 0;
	srand(2222222);
	char buff[256];
	char *input = buff;
	int numFrames = 0;
	float nms = .4;
	x -= w / 2;
	y -= h / 2;
	int speed = 33;
	int stopflag = 0;
	int key;
	for (i = 0; i < m; i++)
	{
		if (paths[i])
		{
			strncpy(input, paths[i], 256);
		}

		CvCapture  *cap1 = cvCreateFileCapture(input);
		//cap = cvCaptureFromFile(input);
		if (!cap1)
		{
			printf("%d  Couldn't load %s.\n", i+1,input);
			continue;
		}
		printf("%d  Load %s\n   Remain %d videos\n", i+1,input,m-i-1);
		numFrames = (int)cvGetCaptureProperty(cap1, CV_CAP_PROP_FRAME_COUNT);
		detection_layer l = net.layers[net.n - 1];
		int j;

		boxes = (box *)calloc(l.side*l.side*l.n, sizeof(box));
		probs = (float **)calloc(l.side*l.side*l.n, sizeof(float *));
		for (j = 0; j < l.side*l.side*l.n; ++j) probs[j] = (float *)calloc(l.classes, sizeof(float *));
		int countFrames = 0;
		int flag = 0;
		IplImage *frame = NULL;
		while (countFrames < numFrames){
			if (flag == 1) break;
			frame = cvQueryFrame(cap1);
			image im = ipl_to_image(frame);
			image im_r = crop_image(im, x, y, w, h);
			image im_s = resize_image(im_r, net.w, net.h);
			countFrames++;
			printf("countFrames=%d,numFrame=%d\n", countFrames, numFrames);//debuging
			float *X = im_s.data;
			float *predictions = network_predict(net, X);
			convert_yolo_detections(predictions, l.classes, l.n, l.sqrt, l.side, 1, 1, demo_thresh, probs, boxes, 0);
			if (nms > 0) do_nms(boxes, probs, l.side*l.side*l.n, l.classes, nms);
			draw_detections(im_r, l.side*l.side*l.n, demo_thresh, boxes, probs, voc_names, voc_labels, 1);
			
			show_image(im_r, "YOLO");
			char savename[256];
			char savename1[256];
			char head_savepath[256];
			char nohead_savepath[256];
			sprintf(head_savepath, "%s/head", &Folder_name);
			sprintf(nohead_savepath, "%s/nohead", &Folder_name);
			sprintf(savename, "%s/%d_%07d.jpg", head_savepath, i, countFrames);
			sprintf(savename1, "%s/%d_%07d.jpg", nohead_savepath, i, countFrames);
			mkdir(&Folder_name, NULL);
			mkdir(head_savepath, NULL);
			mkdir(nohead_savepath, NULL);
			if (stopflag==1) 
				key = cvWaitKey(speed);
			else key = cvWaitKey(0);
			switch (key)
			{		
			case 32:{ stopflag = !stopflag; break; }
			case 'd':{ countFrames -= 2;  if (countFrames < 0) countFrames = 0; cvSetCaptureProperty(cap1, CV_CAP_PROP_POS_FRAMES, countFrames); break; }
			case 'D':{ countFrames -= 2;  if (countFrames < 0) countFrames = 0; cvSetCaptureProperty(cap1, CV_CAP_PROP_POS_FRAMES, countFrames); break; }
			case 'a':{ save_image_jpg(im, savename1); break; }
			case 'A':{ save_image_jpg(im, savename1); break; }
			case 's':{ save_image_jpg(im, savename); break; }
			case 'S':{ save_image_jpg(im, savename); break; }
			case 'o':{ speed +=10; break; }
			case 'i':{ speed -= 10; break; }
			case 'l':{ countFrames += 5; if (countFrames < numFrames) cvSetCaptureProperty(cap1, CV_CAP_PROP_POS_FRAMES, countFrames); break; }
			case 'L':{ countFrames += 5; if (countFrames < numFrames) cvSetCaptureProperty(cap1, CV_CAP_PROP_POS_FRAMES, countFrames); break; }
			case  27:{ flag=1; break; }
			default:{}
			}
			free_image(im_s);
			free_image(im_r);
			free_image(im);
		}
		cvReleaseCapture(&cap1);
	}
}
void demo_sort_fold(char *cfgfile, char *weightfile, char *filename, char *cfgfile1, char *weightfile1, float thresh, int cam_index)
{
	demo_thresh = thresh;
	printf("YOLO demo\n");
	net = parse_network_cfg(cfgfile);
	if (weightfile){
		load_weights(&net, weightfile);
	}
	set_batch_network(&net, 1);
	net1 = parse_network_cfg(cfgfile1);
	if (weightfile1){
		load_weights(&net1, weightfile1);
	}
	set_batch_network(&net1, 1);
	list *plist = get_paths(filename);
	char **paths = (char **)list_to_array(plist);
	int m = plist->size;
	int i = 0;
	srand(2222222);
	char buff[256];
	char *input = buff;
	int numFrames = 0;
	float nms = .4;
	for (i = 0; i < m; i++)
	{
		if (paths[i])
		{
			strncpy(input, paths[i], 256);
		}

		CvCapture  *cap1 = cvCreateFileCapture(input);
		//cap = cvCaptureFromFile(input);
		if (!cap1) error("Couldn't connect to webcam.\n");

		numFrames = (int)cvGetCaptureProperty(cap1, CV_CAP_PROP_FRAME_COUNT);
		detection_layer l = net.layers[net.n - 1];
		int j;

		boxes = (box *)calloc(l.side*l.side*l.n, sizeof(box));
		probs = (float **)calloc(l.side*l.side*l.n, sizeof(float *));
		for (j = 0; j < l.side*l.side*l.n; ++j) probs[j] = (float *)calloc(l.classes, sizeof(float *));
		int countFrames = 0;
		IplImage *frame = NULL;
		while (countFrames < numFrames){
			/*clock_t time;*/
			frame = cvQueryFrame(cap1);
			image im = ipl_to_image(frame);
			image im_s = resize_image(im, net.w, net.h);
			countFrames++;

			float *X = im_s.data;
			float *predictions = network_predict(net, X);
			convert_yolo_detections(predictions, l.classes, l.n, l.sqrt, l.side, 1, 1, demo_thresh, probs, boxes, 0);
			if (nms > 0) do_nms(boxes, probs, l.side*l.side*l.n, l.classes, nms);
			//draw_detections(im, l.side*l.side*l.n, demo_thresh, boxes, probs, voc_names, voc_labels, 1);
			draw_detections_net(&net1, im, l.side*l.side*l.n, demo_thresh, boxes, probs, voc_names, voc_labels, 1);
			/*time = clock();
			printf("Predicted in %f seconds.\n", sec(clock() - time));*/
			printf("countFrames=%d,numFrame=%d\n", countFrames, numFrames);//debuging
			show_image(im, "YOLO");
			cvWaitKey(1);
			free_image(im_s);
			free_image(im);
		}
		cvReleaseCapture(&cap1);
	}
}
#else

void demo_yolo(char *cfgfile, char *weightfile, float thresh, int cam_index, char *filename){
	fprintf(stderr, "YOLO demo needs OpenCV for webcam images.\n");
}
void demo_yolo_fold(char *cfgfile, char *weightfile, float thresh, int cam_index, char *filename){
	fprintf(stderr, "YOLO demo_yolo_fold needs OpenCV for webcam images.\n");
}
void demo_sort_fold(char *cfgfile, char *weightfile, char *filename, char *cfgfile1, char *weightfile1, float thresh, int cam_index){
	fprintf(stderr, "YOLO demo_sort_fold needs OpenCV for webcam images.\n");
}
void demo_yolo_fold2(char *cfgfile, char *weightfile, float thresh, int cam_index, char *filename, int x, int y, int w, int h){
	fprintf(stderr, "YOLO demo_yolo_fold2 needs OpenCV for webcam images.\n");
}

#endif


