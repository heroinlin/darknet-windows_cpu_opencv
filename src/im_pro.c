#include"im_pro.h"
#define MAX 100
void make_random_box(char* image_list_path, char *label_save_path,int box_num)
{
	int channels = 3;
	list *plist = get_paths(image_list_path);
	//int N = plist->size;
	char **paths = (char **)list_to_array(plist);
	int m = plist->size;
        int i;
	for (i = 0; i < m; i++)
	{
		if (!paths[i])
		{
			printf("load error!\n");
			break;
		}
		image im = load_image(paths[i], 0, 0, channels);
		//printf("load %s", paths[i]);
		int j = 0;
		char labels_path[256];
		printf("save to %s/1_%07d.txt\n", label_save_path, i + 1);
		sprintf(labels_path, "%s/1_%07d.txt", label_save_path, i + 1);
		FILE * fp;
		fp = fopen(labels_path, "w+");
		while (j < box_num)
		{
			j++;
			float x = rand() % MAX*0.005 + 0.3;
			float y = rand() % MAX*0.005 + 0.3;
			float w = rand() % MAX*0.005 + 0.3;
			float h = rand() % MAX*0.005 + 0.3;
			float x1 = (1 - x) > x ? x : (1 - x);
			float y1 = (1 - y) > y ? y : (1 - y);
			if (w > 2 * x1)  w = 2 * x1 - 0.01;
			if (h > 2 * y1) h = 2 * y1 - 0.01;
			box box = { x, y, w, h };
			fprintf(fp, "1 %f %f %f %f\n", x, y, w, h);
			//char image_path[256];
			//sprintf(image_path, "%s/%07d", image_save_path,box_num*(i+1)-(box_num-j));
			//im = crop_image(im, (x - 0.5*w) *im.w, (y - 0.5*h)*im.h, w*im.w, h*im.h);
			//save_image_jpg(im, save_path);
		}
		fclose(fp);
		free_image(im);
	}
}
int crop_box2jpg(char* image_list_path, char *image_save_path)
{
	int channels = 3;
	list *plist = get_paths(image_list_path);
	//int N = plist->size;
	char **paths = (char **)list_to_array(plist);
	int m = plist->size;
	int count_label = 0;
        int i;
	for (i = 0; i < m; i++)
	{
		if (!paths[i])
		{
			printf("load error!\n");
			break;
		}
		image im = load_image(paths[i], 0, 0, channels);
		//printf("load %s\n", paths[i]);
//		char *labelpath = find_replace(paths[i], "images", "labels");
//		labelpath = find_replace(labelpath, "JPEGImages", "labels");
//		labelpath = find_replace(labelpath, ".jpg", ".txt");
//		labelpath = find_replace(labelpath, ".JPEG", ".txt");
        char labelpath[256];
        char *labelpath1 = "/home/heroin/workspace/Test/yolo_cpu/bin/Release/results/labels/";
        sprintf(labelpath,"%s%07d.txt",labelpath1,i+1);
		int count = 0;
		box_label *boxes = read_boxes(labelpath, &count);
		float x, y, w, h;
                int j;
		for (j = 0; j < count; j++)
		{
			count_label++;
			x = boxes[j].x;
			y = boxes[j].y;
			w = boxes[j].w * 1.4;//À©Žó¿í
			h = boxes[j].h * 1.4;
			if ((x + w / 2) > 1 || (x - w / 2 )< 0) w = boxes[j].w;
			if ((y + h / 2 )> 1 || (y - h / 2 )< 0) h = boxes[j].h;
			box box = { x, y, w, h };
			char image_path[256];
			sprintf(image_path, "%s/%07d.jpg", image_save_path, count_label);
			//show_image(im, save_path);
			image crop_im = crop_image(im, (x - 0.5*w) *im.w, (y - 0.5*h)*im.h, w*im.w, h*im.h);
			printf("saving to %s \n", image_path);
			save_image_jpg(crop_im, image_path);
			free_image(crop_im);
		}
		free_image(im);
	}
}
void crop_labelbox2jpg(char* image_list_path, char *image_save_path, char *label_save_path, int cx, int cy, int cw, int ch)
{
	int channels = 3;
	list *plist = get_paths(image_list_path);
	//int N = plist->size;
	char **paths = (char **)list_to_array(plist);
	int m = plist->size;
	int count = 0;
	int i;
	for (i = 0; i < m; i++)
	{
		if (!paths[i])
		{
			printf("load error!\n");
			break;
		}
		image im = load_image(paths[i], 0, 0, channels);
		printf("load %s\n", paths[i]);
		char *labelpath = find_replace(paths[i], ".jpg", ".txt");
		//char *labelpath = find_replace(paths[i], "images", "labels");
		//labelpath = find_replace(labelpath, "JPEGImages", "labels");
		//labelpath = find_replace(labelpath, ".jpg", ".txt");
		labelpath = find_replace(labelpath, ".JPEG", ".txt");

		box_label *boxes = read_boxes(labelpath, &count);
		float x, y, w, h;
		int j;
		int flag = 0;
		char labels_path[32];
		sprintf(labels_path, "%s/%s.txt", label_save_path, get_basename(labelpath));

		char image_path[256];
		sprintf(image_path, "%s/%s.jpg", image_save_path, get_basename(paths[i]));
		//show_image(im, save_path);
		image crop_im = crop_image(im, cx, cy, cw, ch);

		float ccx = (float)cx / (float)im.w;
		float ccy = (float)cy / (float)im.h;
		float ccw = (float)cw / (float)im.w;
		float cch = (float)ch / (float)im.h;

		FILE * fp;
		fp = fopen(labels_path, "w+");
		for (j = 0; j < count; j++)
		{
			x = boxes[j].x;
			y = boxes[j].y;
			w = boxes[j].w;
			h = boxes[j].h;
			float x1 = (x - w / 2 - ccx) / ccw;
			float y1 = (y - h / 2 - ccy) / cch;
			float x2 = (x + w / 2 - ccx) / ccw;
			float y2 = (y + h / 2 - ccy) / cch;
			if (x1<0 || y1<0 || x2>1 || y2>1) { flag = 1; break; }
			x = (x - ccx) / ccw;
			y = (y - ccy) / cch;
			w = w / ccw;
			h = h / cch;
			fprintf(fp, "0 %f %f %f %f\n", x, y, w, h);
		}
		fclose(fp);
		if (flag == 0)
		{
			printf("save to %s/%s.txt\n", label_save_path, get_basename(labelpath));
			printf("saving to %s \n", image_path);
			image re_im=resize_image(crop_im, 320, 288);
			save_image_jpg(re_im, image_path);
			free_image(re_im);
		}
		else
		{
			remove(labels_path);
		}
		free_image(crop_im);
		free_image(im);
	}
}
void image_rename(char* image_list_path, char *image_save_path)
{
	list *plist = get_paths(image_list_path);
	//int N = plist->size;
	char **paths = (char **)list_to_array(plist);
	int m = plist->size;
        int i;
	for (i = 0; i < m; i++)
	{
		if (!paths[i])
		{
			printf("load error!\n");
			break;
		}
		char new_path[256];
		sprintf(new_path,"%s/1_%07d.jpg",image_save_path,i+1);
		rename( paths[i], new_path);
	}
}
void image_check(char* image_list_path,int w,int h)
{
    int channels = 3;
	list *plist = get_paths(image_list_path);
	//int N = plist->size;
	char **paths = (char **)list_to_array(plist);
	int m = plist->size;
	int count = 1;
	int flag=0;
	int i;
	char Folder_name[256];
	printf("Please Enter the Folder to save pictures:\n");
	scanf("%s/n",&Folder_name);
	mkdir(&Folder_name, NULL);
	for (i = 0; i < m; i++)
	{
	    if (flag == 1) break;
		if (!paths[i])
		{
			printf("load error!\n");
			break;
		}
		image orig_im = load_image(paths[i], 0, 0, channels);
		//printf("load %s\n", paths[i]);
		char *labelpath = find_replace(paths[i], ".jpg", ".txt");
		//char *labelpath = find_replace(paths[i], "images", "labels");
		//labelpath = find_replace(labelpath, "JPEGImages", "labels");
		//labelpath = find_replace(labelpath, ".jpg", ".txt");
		labelpath = find_replace(labelpath, ".JPEG", ".txt");

		box_label *boxes = read_boxes(labelpath, &count);
		image im=resize_image(orig_im,w,h);
		float x, y, w, h;
		int j;
		int flag = 0;
		//show_image(im, save_path);

		for (j = 0; j < count; j++)
		{
			x = boxes[j].x;
			y = boxes[j].y;
			w = boxes[j].w;
			h = boxes[j].h;
			float x1 = (x - w / 2) *im.w;
			float y1 = (y - h / 2 ) *im.h;
			float x2 = (x + w / 2) *im.w;
			float y2 = (y + h / 2 ) *im.h;
			draw_box_width(im,x1,y1,x2,y2,3,0,1,0);
		}
        char savename[256];
		sprintf(savename, "%s/%07d.jpg", &Folder_name, i);
		char save_labelname[256];
		sprintf(save_labelname, "%s/%07d.txt", &Folder_name, i);
		/*src=cvCreateImage(size, frame->depth, frame->nChannels);
		cvResize(frame, src, CV_INTER_CUBIC);*/
		cvNamedWindow("check", CV_WINDOW_NORMAL);
		show_image(im,"check");
		int key=cvWaitKey(0);
        switch (key)
		{
		//case 's':{ save_image_jpg(im, savename); CopyFileA(labelpath, save_labelname, FALSE); break; }
		case 's':{ save_image_jpg(orig_im, savename); CopyFileA(labelpath, save_labelname, FALSE); break; }
		case 'S':{ save_image_jpg(orig_im, savename); CopyFileA(labelpath, save_labelname, FALSE); break; }
		case 27:{ flag=1; break; }
		default:{}
		}
        free_image(im);
        free_image(orig_im);
		}
	cvDestroyAllWindows();
}
void image_check1(char* image_list_path, int w, int h)
{
	int channels = 3;
	list *plist = get_paths(image_list_path);
	//int N = plist->size;
	char **paths = (char **)list_to_array(plist);
	int m = plist->size;
	int count = 1;
	int flag = 0;
	int i;
	for (i = 0; i < m; i++)
	{
		if (flag == 1) break;
		if (!paths[i])
		{
			printf("load error!\n");
			break;
		}
		image orig_im = load_image(paths[i], 0, 0, channels);
		//printf("load %s\n", paths[i]);
		char *labelpath = find_replace(paths[i], ".jpg", ".txt");
		//char *labelpath = find_replace(paths[i], "images", "labels");
		//labelpath = find_replace(labelpath, "JPEGImages", "labels");
		//labelpath = find_replace(labelpath, ".jpg", ".txt");
		labelpath = find_replace(labelpath, ".JPEG", ".txt");

		box_label *boxes = read_boxes(labelpath, &count);
		image im = resize_image(orig_im, w, h);
		float x, y, w, h;
		int j;
		int flag = 0;
		//show_image(im, save_path);

		for (j = 0; j < count; j++)
		{
			x = boxes[j].x;
			y = boxes[j].y;
			w = boxes[j].w;
			h = boxes[j].h;
			float x1 = (x - w / 2) *im.w;
			float y1 = (y - h / 2) *im.h;
			float x2 = (x + w / 2) *im.w;
			float y2 = (y + h / 2) *im.h;
			draw_box_width(im, x1, y1, x2, y2, 3, 0, 1, 0);
		}
		/*src=cvCreateImage(size, frame->depth, frame->nChannels);
		cvResize(frame, src, CV_INTER_CUBIC);*/
		cvNamedWindow("check", CV_WINDOW_NORMAL);
		show_image(im, "check");
		int key = cvWaitKey(0);
		switch (key)
		{
		case 'd':{ remove(paths[i]); remove(labelpath); break; }//不能退，由于原文件已被删，路径读取无文件会出错
		case 'D':{ remove(paths[i]); remove(labelpath); break; }
		case 27:{ flag = 1; break; }
		default:{}
		}
		free_image(im);
		free_image(orig_im);
	}
	cvDestroyAllWindows();
}
void image_pick(char* image_list_path, int x,int y,int w, int h)
{
	int channels = 3;
	list *plist = get_paths(image_list_path);
	//int N = plist->size;
	char **paths = (char **)list_to_array(plist);
	int m = plist->size;
	int count = 1;
	int flag = 0;
	int i;
	char Folder_name[256];
	printf("Please Enter the Folder to save pictures:\n");
	scanf("%s/n", &Folder_name);
	char savename[256];
	char savename1[256];
	char head_savepath[256];
	char nohead_savepath[256];
	sprintf(head_savepath, "%s/head", &Folder_name);
	sprintf(nohead_savepath, "%s/nohead", &Folder_name);
	mkdir(&Folder_name, NULL);
	mkdir(head_savepath, NULL);
	mkdir(nohead_savepath, NULL);
	for (i = 0; i < m; i++)
	{
		if (flag == 1) break;
		if (!paths[i])
		{
			printf("load error!\n");
			break;
		}
		image orig_im = load_image(paths[i], 0, 0, channels);
		printf("load image %d , remain %d images\n", i+1,m-i-1);
		image crop_im = crop_image(orig_im, x, y, 320, 288);
		image im = resize_image(crop_im, w, h);
		//image im = resize_image(orig_im, w, h);
		
		int flag = 0;
		//show_image(im, save_path);
		char savename[256];
		sprintf(savename, "%s/%07d.jpg", head_savepath, count++);
		char savename1[256];
		sprintf(savename1, "%s/%07d.jpg", nohead_savepath, count++);
		show_image(im, "pick");
		int key = cvWaitKey(0);
		switch (key)
		{
		case 'd':{i -= 2; if (i < 0) i = 0; break; }
		case 'D':{i -= 2; if (i < 0) i = 0; break; }
		case 'a':{ save_image_jpg(orig_im, savename1); break; }
		case 'A':{ save_image_jpg(orig_im, savename1); break; }
		case 's':{ save_image_jpg(orig_im, savename); break; }
		case 'S':{ save_image_jpg(orig_im, savename); break; }
		case 27:{ flag = 1; break; }
		default:{}
		}
		free_image(im);
		free_image(crop_im);
		free_image(orig_im);
	}
}
void make_file(char* image_list_path, char *label_save_path)
{
list *plist = get_paths(image_list_path);
	//int N = plist->size;
	char **paths = (char **)list_to_array(plist);
	int m = plist->size;
        int i;
	for (i = 0; i < m; i++)
	{
	    char labels_path[256];
		printf("save to %s/%07d.txt\n", label_save_path, i + 1);
		sprintf(labels_path, "%s/%07d.txt", label_save_path, i + 1);
		FILE * fp;
		fp = fopen(labels_path, "w+");
		fclose(fp);
}
}
void make_file1(char* image_list_path)
{
	list *plist = get_paths(image_list_path);
	//int N = plist->size;
	char **paths = (char **)list_to_array(plist);
	int m = plist->size;
	int i;
	for (i = 0; i < m; i++)
	{
		char *labels_path = find_replace(paths[i], "jpg", "txt");
		FILE * fp;
		fp = fopen(labels_path, "w+");
		fclose(fp);
	}
}
void crop_jpgs(char* image_list_path,int x, int y, int w, int h)
{
	int channels = 3;
	list *plist = get_paths(image_list_path);
	//int N = plist->size;
	char **paths = (char **)list_to_array(plist);
	int m = plist->size;
	int count_label = 0;
	int i;
	for (i = 0; i < m; i++)
	{
		if (!paths[i])
		{
			printf("load error!\n");
			break;
		}
		image im = load_image(paths[i], 0, 0, channels);
		image crop_im = crop_image(im, x,y,w,h);
		save_image_jpg(crop_im, paths[i]);
		free_image(crop_im);
		free_image(im);
	}
}

static IplImage *src,*dst;
static box boxs;
static char labelname[256];
void on_mouse(int event, int x, int y, int flags, void* ustc)
{
	static CvPoint pre_pt = { -1, -1 };
	static CvPoint cur_pt = { -1, -1 };
	CvFont font;
	cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 0.5, 0.5, 0, 1, CV_AA);
	char temp[16];

	if (event == CV_EVENT_LBUTTONDOWN)
	{
		cvCopy(dst, src,0);
		if (x < 0) x = 0;
		if (y < 0) y = 0;
		if (x > src->width) x = src->width;
		if (y > src->height) y =src->height;
		sprintf(temp, "(%d,%d)", x, y);
		pre_pt = cvPoint(x, y);
		//cvPutText(src, temp, pre_pt, &font, cvScalar(0, 0, 0, 255));
		//cvCircle(src, pre_pt, 3, cvScalar(255, 0, 0, 0), CV_FILLED, CV_AA, 0);
		cvShowImage("src", src);
		cvCopy(src, dst,0);
	}
	else if (event == CV_EVENT_MOUSEMOVE && !(flags & CV_EVENT_FLAG_LBUTTON))
	{
		cvCopy(dst, src,0);
		if (x < 0) x = 0;
		if (y < 0) y = 0;
		if (x > src->width) x = src->width;
		if (y > src->height) y = src->height;
		sprintf(temp, "(%d,%d)", x, y);
		cur_pt = cvPoint(x, y);
		//cvPutText(src, temp, cur_pt, &font, cvScalar(0, 0, 0, 255));
		cvLine(src, cvPoint(x, 0), cvPoint(x, src->height), cvScalar(255, 0, 0, 0), 1, 4, 0);
		cvLine(src, cvPoint(0, y), cvPoint(src->width, y), cvScalar(255, 0, 0, 0), 1, 4, 0);
		cvShowImage("src", src);
	}
	else if (event == CV_EVENT_MOUSEMOVE && (flags & CV_EVENT_FLAG_LBUTTON))
	{
		cvCopy(dst, src,0);
		if (x < 0) x = 0;
		if (y < 0) y = 0;
		if (x > src->width) x = src->width;
		if (y > src->height) y = src->height;
		sprintf(temp, "(%d,%d)", x, y);
		cur_pt = cvPoint(x, y);
		//cvPutText(src, temp, cur_pt, &font, cvScalar(0, 0, 0, 255));
		if (fabs(pre_pt.x - cur_pt.x) > 10 && fabs(pre_pt.y - cur_pt.y) > 10)
		{
			cvRectangle(src, pre_pt, cur_pt, cvScalar(0, 255, 0, 0), 1, 8, 0);
		}
		cvShowImage("src", src);
	}
	else if (event == CV_EVENT_LBUTTONUP)
	{
		if (x < 0) x = 0;
		if (y < 0) y = 0;
		if (x > src->width) x = src->width;
		if (y > src->height) y = src->height;
		sprintf(temp, "(%d,%d)", x, y);
		cur_pt = cvPoint(x, y);
		//cvPutText(src, temp, cur_pt, &font, cvScalar(0, 0, 0, 255));
		//cvCircle(src, cur_pt, 3, cvScalar(255, 0, 0, 0), CV_FILLED, CV_AA, 0);
		if (fabs(pre_pt.x - cur_pt.x) > 10 && fabs(pre_pt.y - cur_pt.y) > 10)
		{
			cvRectangle(src, pre_pt, cur_pt, cvScalar(0, 255, 0, 0), 1, 8, 0);
		}
		cvShowImage("src", src);
		cvCopy(src, dst,0);	
		float width = fabs(pre_pt.x - cur_pt.x);
		float height = fabs(pre_pt.y - cur_pt.y);
		CvRect rect;
		if (pre_pt.x < cur_pt.x && pre_pt.y<cur_pt.y)
		{
			rect = cvRect(pre_pt.x, pre_pt.y, width, height);
		}
		else if (pre_pt.x>cur_pt.x && pre_pt.y<cur_pt.y)
		{
			rect = cvRect(cur_pt.x, pre_pt.y, width, height);
		}
		else if (pre_pt.x>cur_pt.x && pre_pt.y > cur_pt.y)
		{
			rect = cvRect(cur_pt.x, cur_pt.y, width, height);
		}
		else if (pre_pt.x<cur_pt.x && pre_pt.y>cur_pt.y)
		{
			rect = cvRect(pre_pt.x, cur_pt.y, width, height);
		}
		boxs.x = (float)(pre_pt.x + cur_pt.x) / 2.0;
		boxs.y = (float)(pre_pt.y + cur_pt.y) / 2.0;
		boxs.w = width;
		boxs.h = height;
		if (boxs.w > 10 && boxs.h > 10)
		{
			FILE *fps;
			fps = fopen(labelname, "a+");
			fprintf(fps, "0 %f %f %f %f\n", boxs.x / src->width, boxs.y / src->height, boxs.w / src->width, boxs.h / src->height);
			fclose(fps);
		}
	}
}
IplImage* ResizeImage(IplImage *src,float scale)
{
	// allocate memory for the dsc    
	IplImage* dsc = cvCreateImage(cvSize(src->width*scale, src->height*scale),
		src->depth, src->nChannels);

	// resizes Image(input array is resized to fit the output array )   
	cvResize(src, dsc, CV_INTER_LINEAR);
	return dsc;

}
void anno_video_list(char *filename, float scale)
{
	char Folder_name[256];
	printf("Please Enter the Folder to save pictures:\n");
	scanf("%s/n", &Folder_name);
	list *plist = get_paths(filename);
	char **paths = (char **)list_to_array(plist);
	int m = plist->size;
	int i;
	char buff[256];
	char *input = buff;
	int numFrames = 0;
	float nms = .4;
	int speed = 33;
	int stopflag = 0;
	int key;
	//CvSize size;
	//{
	//	size.width = weight, size.height = height;
	//}
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
			printf("%d  Couldn't load %s.\n", i + 1, input);
			continue;
		}
		printf("%d  Load %s\n   Remain %d videos\n", i + 1, input, m - i - 1);
		numFrames = (int)cvGetCaptureProperty(cap1, CV_CAP_PROP_FRAME_COUNT);
		int j;
		int countFrames = 0;
		int flag = 0;
		IplImage *frame = cvCreateImage(cvSize(320, 288), IPL_DEPTH_8U, 1);
		while (countFrames < numFrames){
			if (flag == 1)  break; 
			frame = cvQueryFrame(cap1);
			src = ResizeImage(frame,scale);
			/*src=cvCreateImage(size, frame->depth, frame->nChannels);
			cvResize(frame, src, CV_INTER_CUBIC);*/
			dst = cvCloneImage(src);
			cvNamedWindow("src", CV_WINDOW_NORMAL);
			cvSetMouseCallback("src", on_mouse, 0);

			cvShowImage("src", src);
			
			countFrames++;
			printf("countFrames=%d,numFrame=%d\n", countFrames, numFrames);//debuging
		
			char savename[256];
			char savename1[256];
			char head_savepath[256];
			char nohead_savepath[256];
			sprintf(head_savepath, "%s/head", &Folder_name);
			sprintf(nohead_savepath, "%s/nohead", &Folder_name);
			mkdir(&Folder_name, NULL);
			mkdir(head_savepath, NULL);
			mkdir(nohead_savepath, NULL);
			sprintf(savename, "%s/%d_%07d.jpg", head_savepath, i, countFrames);
			sprintf(savename1, "%s/%d_%07d.jpg", nohead_savepath, i, countFrames);
			sprintf(labelname, "%s/%d_%07d.txt", head_savepath, i, countFrames);
			if (stopflag == 1)
				key = cvWaitKey(speed);
			else key = cvWaitKey(0);
			switch (key)
			{
			case 32:{ stopflag = !stopflag; break; }
			case 'c':{ remove(labelname); countFrames -= 1; cvSetCaptureProperty(cap1, CV_CAP_PROP_POS_FRAMES, countFrames); break; }
			case 'C':{ remove(labelname); countFrames -= 1;  cvSetCaptureProperty(cap1, CV_CAP_PROP_POS_FRAMES, countFrames); break; }
			case 'd':{ countFrames -= 2; if (countFrames < 0)countFrames = 0; cvSetCaptureProperty(cap1, CV_CAP_PROP_POS_FRAMES, countFrames); break; }
			case 'D':{ countFrames -= 2; if (countFrames < 0)countFrames = 0;  cvSetCaptureProperty(cap1, CV_CAP_PROP_POS_FRAMES, countFrames); break; }
			case 's':{ cvSaveImage(savename,frame,0); break; }
			case 'S':{ cvSaveImage(savename,frame, 0); break; }
			case 'a':{ cvSaveImage(savename1, frame, 0); break; }
			case 'A':{ cvSaveImage(savename1, frame, 0); break; }
			case 'o':{ speed += 10; break; }
			case 'i':{ speed -= 10; break; }
			case 'q':{ flag = 1; i -= 2; break; }
			case 'Q':{ flag = 1; i -= 2; break; }
			case 'l':{ countFrames += 5; if (countFrames < numFrames) cvSetCaptureProperty(cap1, CV_CAP_PROP_POS_FRAMES, countFrames); break; }
			case 'L':{ countFrames += 5; if (countFrames < numFrames) cvSetCaptureProperty(cap1, CV_CAP_PROP_POS_FRAMES, countFrames); break; }
			case 27:{ flag = 1; break; }
			default:{}
			}		
			cvReleaseImage(&src);
			cvReleaseImage(&dst);
		}
		cvReleaseCapture(&cap1);
		
	}
	
	cvDestroyAllWindows();
	
}
void anno_images_list(char *filename, float scale)
{
	char Folder_name[256];
	printf("Please Enter the Folder to save pictures:\n");
	scanf("%s/n", &Folder_name);
	list *plist = get_paths(filename);
	char **paths = (char **)list_to_array(plist);
	int m = plist->size;
	printf("m=%d\n", m);
	int i;
	char buff[256];
	char *input = buff;
	float nms = .4;
	int key;
	int flag = 0;
	//CvSize size;
	//{
	//	size.width = weight, size.height = height;
	//}
	for (i = 0; i < m; i++)
	{
		if (flag==1)	break;
		if (paths[i])
		{
			strncpy(input, paths[i], 256);
		}
		IplImage *images = cvLoadImage(input,0);
		//cap = cvCaptureFromFile(input);
		printf("%d  Load %s\n   Remain %d images\n", i + 1, input, m - i - 1);	
		src = ResizeImage(images, scale);
		/*src=cvCreateImage(size, frame->depth, frame->nChannels);
		cvResize(frame, src, CV_INTER_CUBIC);*/
		dst = cvCloneImage(src);
		cvNamedWindow("src", CV_WINDOW_NORMAL);
		cvSetMouseCallback("src", on_mouse, 0);
		cvShowImage("src", src);
		char savename[256];
		char savename1[256];
		char head_savepath[256];
		char nohead_savepath[256];
		sprintf(head_savepath, "%s/head", &Folder_name);
		sprintf(nohead_savepath, "%s/nohead", &Folder_name);
		mkdir(&Folder_name, NULL);
		mkdir(head_savepath, NULL);
		mkdir(nohead_savepath, NULL);
		sprintf(savename, "%s/%07d.jpg", head_savepath, i);
		sprintf(savename1, "%s/%07d.jpg", nohead_savepath, i);
		sprintf(labelname, "%s/%07d.txt", head_savepath, i);
		key = cvWaitKey(0);
		switch (key)
		{
		case 'c':{ remove(labelname); i -= 1;  break; }
		case 'C':{ remove(labelname); i -= 1;  break; }
		case 'd':{ i -= 2; if (i < 0)i = 0; break; }
		case 'D':{ i -= 2; if (i < 0)i = 0;  break; }
		case 's':{ cvSaveImage(savename, images, 0); break; }
		case 'S':{ cvSaveImage(savename, images, 0); break; }
		case 'a':{ cvSaveImage(savename1, images, 0); break; }
		case 'A':{ cvSaveImage(savename1, images, 0); break; }
		case 27:{ flag = 1; break; }
		default:{}
		}
		cvReleaseImage(&src);
		cvReleaseImage(&dst);
		cvReleaseImage(&images);
		}
	cvDestroyAllWindows();
}

//int main(int argc, char **argv)
//{
//	cvNamedWindow("Annotated", CV_WINDOW_NORMAL);
//	cvResizeWindow("Annotated", 448, 448);
//	char*dir_path = argv[1];
//	//anno_video_folder(dir_path,640,576);
//	window_weight = 720;
//	window_height = 576;
//	anno_image_folder(dir_path, window_weight, window_height);
//	cvDestroyAllWindows();
//	return 0;
//}