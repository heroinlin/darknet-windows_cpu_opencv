#ifndef   IM_PRO_H
#define  IM_PRO_H
#include "utils.h"
#include "box.h"
#include"image.h"
#include"data.h"
#include <windows.h>



#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#include "opencv/cv.h"
#endif
void make_random_box(char* image_list_path, char *label_save_path,int box_num);
int crop_box2jpg(char* image_list_path, char *image_save_path);
void crop_labelbox2jpg(char* image_list_path, char *image_save_path, char *label_save_path, int cx, int cy, int cw, int ch);
void image_rename(char* image_list_path, char *image_save_path);
void image_check(char* image_list_path, int w, int h);
void image_check1(char* image_list_path, int w, int h);
void make_file(char* image_list_path, char *label_save_path);
void crop_jpgs(char* image_list_path, int x, int y, int w, int h);
void anno_video_list(char *filename, float scale);
void anno_images_list(char *filename, float scale);
#endif
