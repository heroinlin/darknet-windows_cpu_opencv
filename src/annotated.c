#include "cv.h"
#include "highgui.h"
#include "box.h"
#include "image.h"
#include <string.h>
#include <windows.h>
#include <stdlib.h>
#include <stdio.h>
//全局变量
int is_drawing = 0;
int box_num = 0;
box *drawing_box;
image img, img1;


static void help();
static void onMouse(int event, int x, int y);

int main(int argc, char** argv)
{
	CvFont font;
	CvScalar scalar;
	char text[10];
	char *txt_path = argv[1];
	char *video_path = argv[2];
	// 初始化字体
	double hScale = 1;
	double vScale = 1;
	int lineWidth = 3;// 相当于写字的线条
	scalar = CV_RGB(255, 0, 0);

	int frame_counter = 0;
	int obj_id = 0;

	CvCapture *capture = cvCreateFileCapture(video_path);
	IplImage *frame = cvQueryFrame(capture);
	img = ipl_to_image(frame);
	img1 =copy_image(img);
	FILE *file;
	file = fopen(txt_path, "w+");
	help();

	for (obj_id=0; obj_id < box_num;obj_id++)
	{
		draw_bbox(img1, drawing_box[obj_id], 1, 0, 1, 0);
	}
	show_image(img,"video");

	cvSetMouseCallback("video", onMouse, 0);

	while (1)
	{
		int c = cvWaitKey(0);
		if ((c & 255) == 27)
		{
			cout << "Exiting ...\n";
			break;
		}

		switch ((char)c)
		{
		case 'n':
			//read the next frame
			++frame_counter;
			frame = cvQueryFrame(capture);
			img = ipl_to_image(frame);
			img1 = copy_image(img);
			if (!frame){
				cout << "\nVideo Finished!" << endl;
				return 0;
			}

			//save all of the labeling rects
			for (obj_id=0; obj_id < box_num; obj_id++)
			{
				draw_bbox(img1, drawing_box[obj_id], 1, 0, 1, 0);
				_itoa(obj_id, text, 10);
				fprintf(file, "%d %d  %f %f %f %f\n", frame_counter, obj_id, drawing_box[obj_id].x, drawing_box[obj_id].y, drawing_box[obj_id].w, drawing_box[obj_id].h);
			}
			obj_id = 0;
			break;
		}
		show_image(img1, "video");
	}
	fclose(file);
	cvNamedWindow("video", 0);
	cvReleaseCapture(&capture);
	cvDestroyWindow("video");
	return 0;
}

static void help()
{
	cout << "This program designed for labeling video \n"
		<< "Coded by L. Wei on 9/4/2013\n" << endl;
	cout << "Use the mouse to draw rectangle on the image for labeling.\n" << endl;
	cout << "Hot keys: \n"
		"\tESC - quit the program\n"
		"\tn - next frame of the video\n"
		"\tc - clear all the labels\n"
		<< endl;
}

static void onMouse(int event, int x, int y)
{
	switch (event)
	{
	case CV_EVENT_LBUTTONDOWN:
		//the left up point of the rect
		is_drawing = 1;
		drawing_box[box_num].x = x;
		drawing_box[box_num].y = y;
		break;
	case CV_EVENT_MOUSEMOVE:
		//adjust the rect (use color blue for moving)
		if (is_drawing){
			drawing_box[box_num].w = x - drawing_box[box_num].x;
			drawing_box[box_num].h = y - drawing_box[box_num].y;
			img1= copy_image(img);
			draw_bbox(img1, drawing_box[box_num],1,1,0,0);
		}
		break;
	case CV_EVENT_LBUTTONUP:
		//finish drawing the rect (use color green for finish)
		if (is_drawing){
			drawing_box[box_num].w = x - drawing_box[box_num].x;
			drawing_box[box_num].h = y - drawing_box[box_num].y;
			img1 = copy_image(img);
			draw_bbox(img1, drawing_box[box_num], 1, 0, 0, 1);
		}
		is_drawing = 0;
		box_num++;
		break;
	}
	show_image(img1, "video");
	return;
}