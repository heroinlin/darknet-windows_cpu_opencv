#include "network.h"
#include "detection_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "im_pro.h"
#include <windows.h>
#include <stdio.h>

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#endif

/* Max class number here */
#define CLASSNUM 1
image voc_labels[CLASSNUM];
static frame_num = 0;

char *voc_names[] = { "person" };

void train_yolo(char *cfgfile, char *backup_directory, char *train_images, char *weightfile)
{
	//char *train_images = "data/voc/train.txt";
	//char *backup_directory = "backup/";

	srand(time(0));
	data_seed = time(0);
	char *base = basecfg(cfgfile);// 输出网络名称
	printf("%s\n", base);
	float avg_loss = -1;
	network net = parse_network_cfg(cfgfile);

	if (weightfile){
		load_weights(&net, weightfile);
	}
	printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
	int imgs = net.batch*net.subdivisions;
	/*（注释）parse_net_options()中，int subdivs = option_find_int(options, "subdivisions",1);net->batch /= subdivs;
	net->batch *= net->time_steps;net->subdivisions = subdivs;也就是说imgs等于cfg文件中batch的值*/
	//printf("imgs=%d,batch=%d，subdivisions=%d\n", imgs,net.batch,net.subdivisions);//debuging
	int i = *net.seen / imgs;//net.seen初始值为0
	data train, buffer;


	layer l = net.layers[net.n - 1];
	int side = l.side;
	int classes = l.classes;
	float jitter = l.jitter;

	list *plist = get_paths(train_images);
	//int N = plist->size;
	char **paths = (char **)list_to_array(plist);

	load_args args = { 0 };
	args.w = net.w;
	args.h = net.h;
	//printf("args.c=%d,net.c=%d\n", args.c, net.c);
	args.c = net.c;
	//printf("args.c=%d,net.c=%d\n", args.c, net.c);
	args.paths = paths;
	args.n = imgs;
	args.m = plist->size;
	args.classes = classes;
	args.jitter = jitter;
	args.num_boxes = side;
	args.d = &buffer;
	//printf("args.c%d,net.c%d/n", args.c, net.c);
	if (args.c == 1)
		args.type = REGION_DATA_GRAY;
	else
		args.type = REGION_DATA;

	printf("Side: %d, Classes: %d, Jitter: %g\n", side, classes, jitter);

	pthread_t load_thread = load_data_in_thread(args);
	clock_t time;
	//while(i*imgs < N*120){
	//printf("test 1  %d\n", net.batch);//debuging
	while (get_current_batch(net) < net.max_batches){
		i += 1;
		time = clock();
		//printf("test 2  %d\n", net.batch);//debuging
		pthread_join(load_thread, 0);
		train = buffer;
		load_thread = load_data_in_thread(args);

		printf("Loaded: %lf seconds\n", sec(clock() - time));
		time = clock();
		//printf("test here1\n");//debuging
		float loss = train_network(net, train);
		//printf("test here2\n");//debuging
		if (avg_loss < 0) avg_loss = loss;
		avg_loss = avg_loss*.9 + loss*.1;
		printf("%d: %f, %f avg, %f rate, %lf seconds, %d images\n", i, loss, avg_loss, get_current_rate(net), sec(clock() - time), i*imgs);
		if (i % 1000 == 0 || i == 600){
			char buff[256];
			sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
			save_weights(net, buff);
		}
		free_data(train);
	}
	char buff[256];
	sprintf(buff, "%s/%s_final.weights", backup_directory, base);
	save_weights(net, buff);
}

void convert_yolo_detections(float *predictions, int classes, int num, int square, int side, int w, int h, float thresh, float **probs, box *boxes, int only_objectness)
{
	int i, j, n;
	//int per_cell = 5*num+classes;
	for (i = 0; i < side*side; ++i){
		int row = i / side;
		int col = i % side;
		for (n = 0; n < num; ++n){
			int index = i*num + n;
			int p_index = side*side*classes + i*num + n;
			float scale = predictions[p_index];
			int box_index = side*side*(classes + num) + (i*num + n) * 4;
			boxes[index].x = (predictions[box_index + 0] + col) / side * w;
			boxes[index].y = (predictions[box_index + 1] + row) / side * h;
			boxes[index].w = pow(predictions[box_index + 2], (square ? 2 : 1)) * w;
			boxes[index].h = pow(predictions[box_index + 3], (square ? 2 : 1)) * h;
			for (j = 0; j < classes; ++j){
				int class_index = i*classes;
				float prob = scale*predictions[class_index + j];
				probs[index][j] = (prob > thresh) ? prob : 0;
			}
			if (only_objectness){
				probs[index][0] = scale;
			}
		}
	}
}

void print_yolo_detections(FILE **fps, char *id, box *boxes, float **probs, int total, int classes, int w, int h)
{
	int i, j;
	for (i = 0; i < total; ++i){
		float xmin = boxes[i].x - boxes[i].w / 2.;
		float xmax = boxes[i].x + boxes[i].w / 2.;
		float ymin = boxes[i].y - boxes[i].h / 2.;
		float ymax = boxes[i].y + boxes[i].h / 2.;

		if (xmin < 0) xmin = 0;
		if (ymin < 0) ymin = 0;
		if (xmax > w) xmax = w;
		if (ymax > h) ymax = h;

		for (j = 0; j < classes; ++j){
			if (probs[i][j]) fprintf(fps[j], "%s %f %f %f %f %f\n", id, probs[i][j],
				xmin, ymin, xmax, ymax);
		}
	}
}

void validate_yolo(char *cfgfile, char *weightfile)
{
	network net = parse_network_cfg(cfgfile);
	if (weightfile){
		load_weights(&net, weightfile);
	}
	set_batch_network(&net, 1);
	fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
	srand(time(0));

	char *base = "results/comp4_det_test_";
	list *plist = get_paths("data/voc.2007.test");
	//list *plist = get_paths("data/voc.2012.test");
	char **paths = (char **)list_to_array(plist);

	layer l = net.layers[net.n - 1];
	int classes = l.classes;
	int square = l.sqrt;
	int side = l.side;

	int j;
	FILE **fps = calloc(classes, sizeof(FILE *));
	for (j = 0; j < classes; ++j){
		char buff[1024];
		snprintf(buff, 1024, "%s%s.txt", base, voc_names[j]);
		fps[j] = fopen(buff, "w");
	}
	box *boxes = calloc(side*side*l.n, sizeof(box));
	float **probs = calloc(side*side*l.n, sizeof(float *));
	for (j = 0; j < side*side*l.n; ++j) probs[j] = calloc(classes, sizeof(float *));

	int m = plist->size;
	int i = 0;
	int t;

	float thresh = .001;
	int nms = 1;
	float iou_thresh = .5;

	int nthreads = 2;
	image *val = calloc(nthreads, sizeof(image));
	image *val_resized = calloc(nthreads, sizeof(image));
	image *buf = calloc(nthreads, sizeof(image));
	image *buf_resized = calloc(nthreads, sizeof(image));
	pthread_t *thr = calloc(nthreads, sizeof(pthread_t));

	load_args args = { 0 };
	args.w = net.w;
	args.h = net.h;
	args.type = IMAGE_DATA;

	for (t = 0; t < nthreads; ++t){
		args.path = paths[i + t];
		args.im = &buf[t];
		args.resized = &buf_resized[t];
		thr[t] = load_data_in_thread(args);
	}
	time_t start = time(0);
	for (i = nthreads; i < m + nthreads; i += nthreads){
		fprintf(stderr, "%d\n", i);
		for (t = 0; t < nthreads && i + t - nthreads < m; ++t){
			pthread_join(thr[t], 0);
			val[t] = buf[t];
			val_resized[t] = buf_resized[t];
		}
		for (t = 0; t < nthreads && i + t < m; ++t){
			args.path = paths[i + t];
			args.im = &buf[t];
			args.resized = &buf_resized[t];
			thr[t] = load_data_in_thread(args);
		}
		for (t = 0; t < nthreads && i + t - nthreads < m; ++t){
			char *path = paths[i + t - nthreads];
			char *id = basecfg(path);
			float *X = val_resized[t].data;
			float *predictions = network_predict(net, X);
			int w = val[t].w;
			int h = val[t].h;
			convert_yolo_detections(predictions, classes, l.n, square, side, w, h, thresh, probs, boxes, 0);
			if (nms) do_nms_sort(boxes, probs, side*side*l.n, classes, iou_thresh);
			print_yolo_detections(fps, id, boxes, probs, side*side*l.n, classes, w, h);
			free(id);
			free_image(val[t]);
			free_image(val_resized[t]);
		}
	}
	fprintf(stderr, "Total Detection Time: %f Seconds\n", (double)(time(0) - start));
}

void validate_yolo_recall(char *cfgfile, char *weightfile)
{
	network net = parse_network_cfg(cfgfile);
	if (weightfile){
		load_weights(&net, weightfile);
	}
	set_batch_network(&net, 1);
	fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
	srand(time(0));

	char *base = "results/comp4_det_test_";
	list *plist = get_paths("data/voc.2007.test");
	char **paths = (char **)list_to_array(plist);

	layer l = net.layers[net.n - 1];
	int classes = l.classes;
	int square = l.sqrt;
	int side = l.side;

	int j, k;
	FILE **fps = calloc(classes, sizeof(FILE *));
	for (j = 0; j < classes; ++j){
		char buff[1024];
		snprintf(buff, 1024, "%s%s.txt", base, voc_names[j]);
		fps[j] = fopen(buff, "w");
	}
	box *boxes = calloc(side*side*l.n, sizeof(box));
	float **probs = calloc(side*side*l.n, sizeof(float *));
	for (j = 0; j < side*side*l.n; ++j) probs[j] = calloc(classes, sizeof(float *));

	int m = plist->size;
	int i = 0;

	float thresh = .001;
	float iou_thresh = .5;
	float nms = 0;

	int total = 0;
	int correct = 0;
	int proposals = 0;
	float avg_iou = 0;

	for (i = 0; i < m; ++i){
		char *path = paths[i];
		image orig = load_image_color(path, 0, 0);
		image sized = resize_image(orig, net.w, net.h);
		char *id = basecfg(path);
		float *predictions = network_predict(net, sized.data);
		convert_yolo_detections(predictions, classes, l.n, square, side, 1, 1, thresh, probs, boxes, CLASSNUM);
		if (nms) do_nms(boxes, probs, side*side*l.n, 1, nms);

		char *labelpath = find_replace(path, "images", "labels");
		labelpath = find_replace(labelpath, "JPEGImages", "labels");
		labelpath = find_replace(labelpath, ".jpg", ".txt");
		labelpath = find_replace(labelpath, ".JPEG", ".txt");

		int num_labels = 0;
		box_label *truth = read_boxes(labelpath, &num_labels);
		for (k = 0; k < side*side*l.n; ++k){
			if (probs[k][0] > thresh){
				++proposals;
			}
		}
		for (j = 0; j < num_labels; ++j) {
			++total;
			box t = { truth[j].x, truth[j].y, truth[j].w, truth[j].h };
			float best_iou = 0;
			for (k = 0; k < side*side*l.n; ++k){
				float iou = box_iou(boxes[k], t);
				if (probs[k][0] > thresh && iou > best_iou){
					best_iou = iou;
				}
			}
			avg_iou += best_iou;
			if (best_iou > iou_thresh){
				++correct;
			}
		}

		fprintf(stderr, "%5d %5d %5d\tRPs/Img: %.2f\tIOU: %.2f%%\tRecall:%.2f%%\n", i, correct, total, (float)proposals / (i + 1), avg_iou * 100 / total, 100.*correct / total);
		free(id);
		free_image(orig);
		free_image(sized);
	}
}

void test_yolo(char *cfgfile, char *weightfile, char *filename, float thresh)
{

	network net = parse_network_cfg(cfgfile);
	if (weightfile){
		load_weights(&net, weightfile);
	}
	detection_layer l = net.layers[net.n - 1];
	set_batch_network(&net, 1);
	srand(2222222);
	clock_t time;
	char buff[256];
	char *input = buff;
	int j;
	float nms = .5;
	box *boxes = calloc(l.side*l.side*l.n, sizeof(box));
	float **probs = calloc(l.side*l.side*l.n, sizeof(float *));
	for (j = 0; j < l.side*l.side*l.n; ++j) probs[j] = calloc(l.classes, sizeof(float *));
	while (1){
		if (filename){
			strncpy(input, filename, 256);
		}
		else {
			printf("Enter Image Path: ");
			fflush(stdout);
			input = fgets(input, 256, stdin);
			if (!input) return;
			strtok(input, "\n");
		}
		int channels = net.c;
		image im = load_image(input, 0, 0, channels);
		image sized = resize_image(im, net.w, net.h);
		float *X = sized.data;
		time = clock();
		float *predictions = network_predict(net, X);
		printf("%s: Predicted in %f seconds.\n", input, sec(clock() - time));
		convert_yolo_detections(predictions, l.classes, l.n, l.sqrt, l.side, 1, 1, thresh, probs, boxes, 0);
		if (nms) do_nms_sort(boxes, probs, l.side*l.side*l.n, l.classes, nms);
		//draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, voc_names, voc_labels, 20);
		//draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, voc_names, 0, CLASSNUM);
		//draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, voc_names, 0, l.classes);
		if (net.c == 1)
			draw_detections_gray(im, l.side*l.side*l.n, thresh, boxes, probs, voc_names, 0, CLASSNUM);
		else
			draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, voc_names, 0, CLASSNUM);

		show_image(im, "predictions");
		save_image(im, "predictions");

		show_image(sized, "resized");
		free_image(im);
		free_image(sized);
#ifdef OPENCV
		cvWaitKey(0);
		cvDestroyAllWindows();
#endif
		if (filename) break;
	}
}
char *get_basename(char* mystr) {
	char *retstr;
	char *lastdot;
	if (mystr == NULL)
		return NULL;
	if ((retstr = malloc(strlen(mystr) + 1)) == NULL)
		return NULL;
	char *basename = strrchr(mystr, '/');
	if (basename == NULL){
		strcpy(retstr, mystr);
	}
	else{
		strcpy(retstr, basename + 1);
	}
	lastdot = strrchr(retstr, '.');
	if (lastdot != NULL)
		*lastdot = '\0';
	return retstr;
}
void test_fold_yolo(char *cfgfile, char *weightfile, char *filename, float thresh)
{

	network net = parse_network_cfg(cfgfile);
	if (weightfile){
		load_weights(&net, weightfile);
	}
	detection_layer l = net.layers[net.n - 1];
	set_batch_network(&net, 1);
	srand(2222222);
	clock_t time;
	char buff[256];
	char *input = buff;
	int j;
	float nms = .5;
	box *boxes = calloc(l.side*l.side*l.n, sizeof(box));
	float **probs = calloc(l.side*l.side*l.n, sizeof(float *));
	for (j = 0; j < l.side*l.side*l.n; ++j) probs[j] = calloc(l.classes, sizeof(float *));
	list *plist = get_paths(filename);
	char **paths = (char **)list_to_array(plist);
	int m = plist->size;
	int i = 0;
	for (i = 0; i < m; i++){
		if (paths[i]){
			strncpy(input, paths[i], 256);
		}
		else {
			printf("Enter Image Path: ");
			fflush(stdout);
			input = fgets(input, 256, stdin);
			if (!input) return;
			strtok(input, "\n");
		}
		int channels = net.c;
		image im = load_image(input, 0, 0, channels);
		image sized = resize_image(im, net.w, net.h);
		float *X = sized.data;
		time = clock();
		float *predictions = network_predict(net, X);
		printf("%s: Predicted in %f seconds.\n", input, sec(clock() - time));
		convert_yolo_detections(predictions, l.classes, l.n, l.sqrt, l.side, 1, 1, thresh, probs, boxes, 0);
		if (nms) do_nms_sort(boxes, probs, l.side*l.side*l.n, l.classes, nms);
		char *basename = get_basename(input);
		if (net.c == 1)
			draw_detections_gray(im, l.side*l.side*l.n, thresh, boxes, probs, voc_names, 0, 1);
		else
			draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, voc_names, 0, 1);
		save_image2result(im, "./results", basename);
		//show_image(sized, "rezise");
		free_image(im);
		free_image(sized);
#ifdef OPENCV
		cvWaitKey(0);
		cvDestroyAllWindows();
#endif
		//    if (paths[i]) break;
	}
}
/*
/*
#ifdef OPENCV
image ipl_to_image(IplImage* src);
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"

void demo_swag(char *cfgfile, char *weightfile, float thresh)
{
network net = parse_network_cfg(cfgfile);
if(weightfile){
load_weights(&net, weightfile);
}
detection_layer layer = net.layers[net.n-1];
CvCapture *capture = cvCaptureFromCAM(-1);
set_batch_network(&net, 1);
srand(2222222);
while(1){
IplImage* frame = cvQueryFrame(capture);
image im = ipl_to_image(frame);
cvReleaseImage(&frame);
rgbgr_image(im);

image sized = resize_image(im, net.w, net.h);
float *X = sized.data;
float *predictions = network_predict(net, X);
draw_swag(im, predictions, layer.side, layer.n, "predictions", thresh);
free_image(im);
free_image(sized);
cvWaitKey(10);
}
}
#else
void demo_swag(char *cfgfile, char *weightfile, float thresh){}
#endif
*/


//#define MAX 100
//void make_random_box(char* image_list_path, char *label_save_path, int box_num)
//{
//	int channels = 3;
//	list *plist = get_paths(image_list_path);
//	//int N = plist->size;
//	char **paths = (char **)list_to_array(plist);
//	int m = plist->size;
//	int i;
//	for (i = 0; i < m; i++)
//	{
//		if (!paths[i])
//		{
//			printf("load error!\n");
//			break;
//		}
//		image im = load_image(paths[i], 0, 0, channels);
//		//printf("load %s", paths[i]);
//		int j = 0;
//		char labels_path[32];
//		printf("save to %s/%07d.txt\n", label_save_path, i + 1);
//		sprintf(labels_path, "%s/%07d.txt", label_save_path, i + 1);
//		FILE * fp;
//		fp = fopen(labels_path, "w+");
//		while (j < box_num)
//		{
//			j++;
//			float x = rand() % MAX*0.005 + 0.3;
//			float y = rand() % MAX*0.005 + 0.3;
//			float w = rand() % MAX*0.005 + 0.3;
//			float h = rand() % MAX*0.005 + 0.3;
//			float x1 = (1 - x) > x ? x : (1 - x);
//			float y1 = (1 - y) > y ? y : (1 - y);
//			if (w > 2 * x1)  w = 2 * x1 - 0.01;
//			if (h > 2 * y1) h = 2 * y1 - 0.01;
//			box box = { x, y, w, h };
//			fprintf(fp, "1 %f %f %f %f\n", x, y, w, h);
//			//char image_path[256];
//			//sprintf(image_path, "%s/%07d", image_save_path,box_num*(i+1)-(box_num-j));
//			//im = crop_image(im, (x - 0.5*w) *im.w, (y - 0.5*h)*im.h, w*im.w, h*im.h);
//			//save_image_jpg(im, save_path);
//		}
//		fclose(fp);
//		free_image(im);
//	}
//}
//void crop_box2jpg(char* image_list_path, char *image_save_path)
//{
//	int channels = 3;
//	list *plist = get_paths(image_list_path);
//	//int N = plist->size;
//	char **paths = (char **)list_to_array(plist);
//	int m = plist->size;
//	int count_label = 0;
//	int i;
//	for (i = 0; i < m; i++)
//	{
//		if (!paths[i])
//		{
//			printf("load error!\n");
//			break;
//		}
//		image im = load_image(paths[i], 0, 0, channels);
//		//printf("load %s\n", paths[i]);
//		char *labelpath = find_replace(paths[i], "images", "labels");
//		labelpath = find_replace(labelpath, "JPEGImages", "labels");
//		labelpath = find_replace(labelpath, ".jpg", ".txt");
//		labelpath = find_replace(labelpath, ".JPEG", ".txt");
//		int count = 0;
//		box_label *boxes = read_boxes(labelpath, &count);
//		float x, y, w, h;
//		int j;
//		for (j = 0; j < count; j++)
//		{
//			count_label++;
//			x = boxes[j].x;
//			y = boxes[j].y;
//			w = boxes[j].w + 0.1;//扩大宽
//			h = boxes[j].h + 0.1;
//			if ((x + w / 2) > 1 || (x - w / 2)< 0) w = boxes[j].w;
//			if ((y + h / 2)> 1 || (y - h / 2)< 0) h = boxes[j].h;
//			box box = { x, y, w, h };
//			char image_path[256];
//			sprintf(image_path, "%s/%07d.jpg", image_save_path, count_label);
//			//show_image(im, save_path);
//			image crop_im = crop_image(im, (x - 0.5*w) *im.w, (y - 0.5*h)*im.h, w*im.w, h*im.h);
//			printf("saving to %s.jpg \n", image_path);
//			save_image_jpg(crop_im, image_path);
//			free_image(crop_im);
//		}
//		free_image(im);
//	}
//}
//void crop_labelbox2jpg(char* image_list_path, char *image_save_path, char *label_save_path, int cx, int cy, int cw, int ch)
//{
//	int channels = 3;
//	list *plist = get_paths(image_list_path);
//	//int N = plist->size;
//	char **paths = (char **)list_to_array(plist);
//	int m = plist->size;
//	int count = 0;
//	int i;
//	for (i = 0; i < m; i++)
//	{
//		if (!paths[i])
//		{
//			printf("load error!\n");
//			break;
//		}
//		image im = load_image(paths[i], 0, 0, channels);
//		//printf("load %s\n", paths[i]);
//		char *labelpath = find_replace(paths[i], ".jpg", ".txt");
//		//char *labelpath = find_replace(paths[i], "images", "labels");
//		//labelpath = find_replace(labelpath, "JPEGImages", "labels");
//		//labelpath = find_replace(labelpath, ".jpg", ".txt");
//		labelpath = find_replace(labelpath, ".JPEG", ".txt");
//
//		box_label *boxes = read_boxes(labelpath, &count);
//		float x, y, w, h;
//		int j;
//		int flag = 0;
//		char labels_path[32];
//		sprintf(labels_path, "%s/%s.txt", label_save_path, get_basename(labelpath));
//
//		char image_path[256];
//		sprintf(image_path, "%s/%s.jpg", image_save_path, get_basename(paths[i]));
//		//show_image(im, save_path);
//		image crop_im = crop_image(im, cx, cy, cw, ch);
//
//		float ccx = (float)cx / (float)im.w;
//		float ccy = (float)cy / (float)im.h;
//		float ccw = (float)cw / (float)im.w;
//		float cch = (float)ch / (float)im.h;
//
//		FILE * fp;
//		fp = fopen(labels_path, "w+");
//		for (j = 0; j < count; j++)
//		{
//			x = boxes[j].x;
//			y = boxes[j].y;
//			w = boxes[j].w;
//			h = boxes[j].h;
//			float x1 = (x - w / 2 - ccx) / ccw;
//			float y1 = (y - h / 2 - ccy) / cch;
//			float x2 = (x + w / 2 - ccx) / ccw;
//			float y2 = (y + h / 2 - ccy) / cch;
//			if (x1<0 || y1<0 || x2>1 || y2>1) { flag = 1; break; }
//			x = (x - ccx) / ccw;
//			y = (y - ccy) / cch;
//			w = w / ccw;
//			h = h / cch;
//			fprintf(fp, "0 %f %f %f %f\n", x, y, w, h);
//		}
//		fclose(fp);
//		if (flag == 0)
//		{
//			printf("save to %s/%s.txt\n", label_save_path, get_basename(labelpath));
//			printf("saving to %s.jpg \n", image_path);
//			resize_image(crop_im, 320, 288);
//			save_image_jpg(crop_im, image_path);
//		}
//		else
//		{
//			remove(labels_path);
//		}
//		free_image(crop_im);
//		free_image(im);
//	}
//}
void draw_detections_net(network *net, image im, int num, float thresh, box *boxes, float **probs, char **names, image *labels, int classes)
{
	int i;
	image temp = im;
	network net1 = *net;
	frame_num++;/**debuging**/
	int box_num = 1;
	for (i = 0; i < num; ++i){
		int class = max_index(probs[i], classes);
		float prob = probs[i][class];
		if (prob > thresh){
			//int width = pow(prob, 1. / 2.) * 10 + 1;
			int width = 3;//the width of box 
			printf("%s: %.2f\n", names[class], prob);
			int offset = class * 17 % classes;
			float red = get_color(0, offset, classes);
			float green = get_color(1, offset, classes);
			float blue = get_color(2, offset, classes);
			float rgb[3];
			rgb[0] = red;
			rgb[1] = green;
			rgb[2] = blue;
			box b = boxes[i];

			int left = (b.x - b.w / 2.)*im.w;
			int right = (b.x + b.w / 2.)*im.w;
			int top = (b.y - b.h / 2.)*im.h;
			int bot = (b.y + b.h / 2.)*im.h;



			if (left < 0) left = 0;
			if (right > im.w - 1) right = im.w - 1;
			if (top < 0) top = 0;
			if (bot > im.h - 1) bot = im.h - 1;
			if (b.w*im.w > 30)
			{
				image crop_im = crop_image(temp, left, top, b.w*im.w, b.h*im.h);
				///**debuging**/
				/*char savename1[256];
				sprintf(savename1, "./results/temp_%d_%d_0", frame_num, box_num++);
				save_image(temp, savename1);
				char savename[256];
				if (classfy_net(crop_im, &net1) == 0)
				{
				sprintf(savename, "./results/crop_im_%d_%d_0", frame_num, box_num++);
				save_image(crop_im, savename);
				draw_box_width(im, left, top, right, bot, width, red, green, blue);
				}
				else
				{
				sprintf(savename, "./results/crop_im_%d_%d_1", frame_num, box_num++);
				save_image(crop_im, savename);
				}*/
				///**debuging**/
				if (classfy_net(crop_im, &net1) == 0) draw_box_width(im, left, top, right, bot, width, red, green, blue);
				if (labels) draw_label(im, top + width, left, labels[class], rgb);
				free_image(crop_im);
			}
		}

	}

	//free_network(net1);
}

int classfy_net(image im, network *net)
{
	network net1 = *net;
	int indexes[2];
	image resize_im = resize_image(im, net1.w, net1.h);
	float *X1 = resize_im.data;
	float *predictions1 = network_predict(net1, X1);
	top_predictions(net1, 2, indexes);
	int i = 0;
	//free_image(im);
	//free_network(net1);
	free_image(resize_im);
	if (predictions1[0] > 0.9999)
	{
		for (i = 0; i < 2; i++){
			int index = indexes[i];
			printf("%d: %f\n", index, predictions1[index]);
		}
		return 0;
	}
	else
		return 1;
}
void sort_yolo(char *cfgfile, char *weightfile, char *cfgfile1, char *weightfile1, char *filename, float thresh)
{

	network net = parse_network_cfg(cfgfile);
	if (weightfile){
		load_weights(&net, weightfile);
	}
	detection_layer l = net.layers[net.n - 1];
	set_batch_network(&net, 1);

	network net1 = parse_network_cfg(cfgfile1);
	if (weightfile1){
		load_weights(&net1, weightfile1);
	}
	set_batch_network(&net1, 1);

	srand(2222222);
	clock_t time;
	char buff[256];
	char *input = buff;
	int j;
	box *boxes = calloc(l.side*l.side*l.n, sizeof(box));
	float **probs = calloc(l.side*l.side*l.n, sizeof(float *));
	for (j = 0; j < l.side*l.side*l.n; ++j) probs[j] = calloc(l.classes, sizeof(float *));
	while (1){
		if (filename){
			strncpy(input, filename, 256);
		}
		else {
			printf("Enter Image Path: ");
			fflush(stdout);
			input = fgets(input, 256, stdin);
			if (!input) return;
			strtok(input, "\n");
		}
		int channels = net.c;
		image im = load_image(input, 0, 0, channels);
		image sized = resize_image(im, net.w, net.h);
		float *X = sized.data;
		time = clock();
		float *predictions = network_predict(net, X);
		printf("%s: Predicted in %f seconds.\n", input, sec(clock() - time));
		convert_yolo_detections(predictions, l.classes, l.n, l.sqrt, l.side, 1, 1, thresh, probs, boxes, 0);
		float x_co = (boxes->x - 0.5 * boxes->w) * sized.w;
		float y_co = (boxes->y - 0.5 * boxes->h) * sized.h;
		float weight = boxes->w*sized.w;
		float height = boxes->h*sized.h;
		image crop_im = crop_image(sized, x_co, y_co, weight, height);
		int indexes[2];
		float *IM = crop_im.data;
		predictions = network_predict(net1, IM);
		top_predictions(net1, 2, indexes);
		if (0 == indexes[0])
		{
			if (net.c == 1)
				draw_detections_gray(im, l.side*l.side*l.n, thresh, boxes, probs, voc_names, 0, CLASSNUM);
			else
				draw_detections_net(&net1, im, l.side*l.side*l.n, thresh, boxes, probs, voc_names, 0, CLASSNUM);
			//draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, voc_names, 0, CLASSNUM);

			show_image(im, "predictions");
			save_image(im, "predictions");

			show_image(sized, "resized");
			free_image(im);
			free_image(sized);
		}
#ifdef OPENCV
		cvWaitKey(0);
		cvDestroyAllWindows();
#endif
		if (filename) break;
	}
}
void sort_fold_yolo(char *cfgfile, char *weightfile, char *cfgfile1, char *weightfile1, char *filename, float thresh)
{

	network net = parse_network_cfg(cfgfile);
	if (weightfile){
		load_weights(&net, weightfile);
	}
	detection_layer l = net.layers[net.n - 1];
	set_batch_network(&net, 1);

	network net1 = parse_network_cfg(cfgfile1);
	if (weightfile1){
		load_weights(&net1, weightfile1);
	}
	set_batch_network(&net1, 1);

	srand(2222222);
	clock_t time;
	char buff[256];
	char *input = buff;
	int j;
	float nms = .5;
	box *boxes = calloc(l.side*l.side*l.n, sizeof(box));
	float **probs = calloc(l.side*l.side*l.n, sizeof(float *));
	for (j = 0; j < l.side*l.side*l.n; ++j) probs[j] = calloc(l.classes, sizeof(float *));
	list *plist = get_paths(filename);
	char **paths = (char **)list_to_array(plist);
	int m = plist->size;
	int i = 0;
	for (i = 0; i < m; i++){
		if (paths[i]){
			strncpy(input, paths[i], 256);
		}
		else {
			printf("Enter Image Path: ");
			fflush(stdout);
			input = fgets(input, 256, stdin);
			if (!input) return;
			strtok(input, "\n");
		}
		int channels = net.c;
		image im = load_image(input, 0, 0, channels);
		image sized = resize_image(im, net.w, net.h);
		float *X = sized.data;
		time = clock();
		float *predictions = network_predict(net, X);
		printf("%s: Predicted in %f seconds.\n", input, sec(clock() - time));
		convert_yolo_detections(predictions, l.classes, l.n, l.sqrt, l.side, 1, 1, thresh, probs, boxes, 0);
		if (nms) do_nms_sort(boxes, probs, l.side*l.side*l.n, l.classes, nms);
		//printf("l.size*l.size*l.n=%d*%d*%d=%d,l.classes=%d\n", l.side, l.side, l.n,l.side*l.side*l.n,l.classes);
		char *basename = get_basename(input);
		if (net.c == 1)
			draw_detections_gray(im, l.side*l.side*l.n, thresh, boxes, probs, voc_names, 0, CLASSNUM);
		else
			draw_detections_net(&net1, im, l.side*l.side*l.n, thresh, boxes, probs, voc_names, 0, CLASSNUM);
		//draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, voc_names, 0, CLASSNUM);
		//show_image(im, "predictions");
		save_image2result(im, "./results", basename);

		//show_image(sized, "resized");
		free_image(im);
		free_image(sized);
#ifdef OPENCV
		cvWaitKey(1);
		cvDestroyAllWindows();
#endif
		//    if (paths[i]) break;
	}
}

void demo_yolo(char *cfgfile, char *weightfile, float thresh, int cam_index, char *filename);
void demo_yolo_fold(char *cfgfile, char *weightfile, float thresh, int cam_index, char *filename);
void demo_sort_fold(char *cfgfile, char *weightfile, char *filename, char *cfgfile1, char *weightfile1, float thresh, int cam_index);
void demo_yolo_fold2(char *cfgfile, char *weightfile, float thresh, int cam_index, char *filename, int x, int y, int w, int h);
//
void run_yolo(int argc, char **argv)
{

	float thresh = find_float_arg(argc, argv, "-thresh", .2);
	int cam_index = find_int_arg(argc, argv, "-c", 0);
	int scale = find_float_arg(argc, argv, "-s", 1.0);
	int x = find_int_arg(argc, argv, "-x", 0);
	int y = find_int_arg(argc, argv, "-y", 0);
	int w = find_int_arg(argc, argv, "-w", 320);
	int h = find_int_arg(argc, argv, "-h", 288);
	if (argc < 4){
		fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)] [synset (optional)]\n", argv[0], argv[1]);
		return;
	}

	char *cfg = argv[3];
	char *weights = (argc > 4) ? argv[4] : 0;
	char *filename = (argc > 5) ? argv[5] : 0;
	char *weights1 = (argc > 6) ? argv[6] : 0;
	char *weights2 = (argc > 7) ? argv[7] : 0;
	if (0 == strcmp(argv[2], "test")) test_yolo(cfg, weights, filename, thresh);
	else if (0 == strcmp(argv[2], "test_fold")) test_fold_yolo(cfg, weights, filename, thresh);
	else if (0 == strcmp(argv[2], "train")) train_yolo(cfg, weights, filename, weights1);
	else if (0 == strcmp(argv[2], "valid")) validate_yolo(cfg, weights);
	else if (0 == strcmp(argv[2], "recall")) validate_yolo_recall(cfg, weights);
	else if (0 == strcmp(argv[2], "demo")) demo_yolo(cfg, weights, thresh, cam_index, filename);
	else if (0 == strcmp(argv[2], "demo_fold")) demo_yolo_fold(cfg, weights, thresh, cam_index, filename);
	else if (0 == strcmp(argv[2], "demo_fold2")) demo_yolo_fold2(cfg, weights, thresh, cam_index, filename, x, y, w, h);
	else if (0 == strcmp(argv[2], "make_box")) make_random_box(cfg, weights, cam_index);
	else if (0 == strcmp(argv[2], "make_file")) make_file1(cfg);
	else if (0 == strcmp(argv[2], "crop_box")) crop_box2jpg(cfg, weights);
	else if (0 == strcmp(argv[2], "crop_jpgs")) crop_jpgs(cfg,x,y,w,h);
	else if (0 == strcmp(argv[2], "crop_label_box")) crop_labelbox2jpg(cfg, weights, filename, x, y, w, h);
	else if (0 == strcmp(argv[2], "pick"))image_pick(cfg,x,y, w, h);
	else if (0 == strcmp(argv[2], "anno")) anno_video_list(cfg, scale); 
	else if (0 == strcmp(argv[2], "anno_image")) anno_images_list(cfg, scale);
	else if (0 == strcmp(argv[2], "check")) image_check(cfg, w,h);
	else if (0 == strcmp(argv[2], "check1")) image_check1(cfg, w, h);
	else if (0 == strcmp(argv[2], "sort")) sort_yolo(cfg, weights, filename, weights1, weights2, thresh);
	else if (0 == strcmp(argv[2], "sort_fold")) sort_fold_yolo(cfg, weights, filename, weights1, weights2, thresh);
	else if (0 == strcmp(argv[2], "demo_sort_fold"))demo_sort_fold(cfg, weights, filename, weights1, weights2, thresh, cam_index);

}
