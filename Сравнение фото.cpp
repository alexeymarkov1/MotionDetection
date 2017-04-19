
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/opencv.hpp>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/videoio/videoio.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "threadpool.h"

//#ifndef min
//#define min(a,b)            (((a) < (b)) ? (a) : (b))
//#endif

using namespace cv;
using namespace std;

VideoCapture cap;
//Mat gray, prevGray, image;
//vector<Point2f> points[2];
TermCriteria termcrit(TermCriteria::COUNT | TermCriteria::EPS, 20, 0.03);
Size subPixWinSize(10, 10), winSize(31, 31);
int countin = 0;
int countout = 0;

IplImage* diff;
IplImage* fimg1;
IplImage* fimg2;


int ThreadsQuantity = 1;

ThreadPool pool(ThreadsQuantity);

void CheckSimil(int l)
{
	int l1 = l*diff->height / ThreadsQuantity;
	int l2 = l1 + diff->height / ThreadsQuantity;
	for (int y = l1; y<l2; y++) {
		uchar* ptr1 = (uchar*)(fimg1->imageData + y * fimg1->widthStep);
		uchar* ptr2 = (uchar*)(fimg2->imageData + y * fimg2->widthStep);
		uchar* ptr = (uchar*)(diff->imageData + y * diff->widthStep);
		for (int x = 0; x<diff->width; x++) {
			// 3 канала:
			if ((abs(ptr1[3 * x] - ptr2[3 * x]) > 70) || (abs(ptr1[3 * x + 1] - ptr2[3 * x + 1]) > 70) || (abs(ptr1[3 * x + 2] - ptr2[3 * x + 2]) > 70)) //&& (((abs(ptr1[3 * x] - ptr2[3 * x])) + (abs(ptr1[3 * x + 1] - ptr2[3 * x + 1])) + (abs(ptr1[3 * x + 2] - ptr2[3 * x + 2]))) > 100))

			{
				ptr[3 * x] = 127;
				ptr[3 * x + 1] = 127;	//Мой вариант
				ptr[3 * x + 2] = 127;
			}

		}
	}
}

void LucasKanade(Mat prevGrey, Mat grey, vector<Point2f> points)
{
	vector<uchar> status;
	vector<float> err;
	vector<Point2f> pointsNew;	
	cornerSubPix(prevGrey, points, subPixWinSize, Size(-1, -1), termcrit);
	calcOpticalFlowPyrLK(prevGrey, grey, points, pointsNew, status, err, winSize, 3, termcrit, 0, 0.001);

	for (int i = 0; i < pointsNew.size(); i++)
	{
		if (!status[i])
			continue;
		if ((int(pointsNew[i].x) > grey.cols / 2) && (int(points[i].x) < grey.cols / 2))
			countin++;
		if ((int(pointsNew[i].x) < grey.cols / 2) && (int(points[i].x) > grey.cols / 2))
			countout++;
		printf("Peresecheno vpravo %d, vlevo %d raz\n", countin, countout);
		//printf("Koord %d: %d %d\n", i, int(points[1][i].x), int(points[1][i].y));
		//circle(image, points[1][i], 3, Scalar(0, 255, 0), -1, 8);
	}
	//printf("Novoe izobr № %d\n", counter);
}
vector<Point2f> showContur(IplImage* img1, IplImage* img2)
{
	// создаём изображения
	//IplImage* hsv1 = cvCreateImage(cvGetSize(img1), 8, 3);
	//IplImage* h_plane = cvCreateImage(cvGetSize(src), 8, 1);
	//IplImage* s_plane = cvCreateImage(cvGetSize(src), 8, 1);
	//IplImage* v_plane = cvCreateImage(cvGetSize(src), 8, 1);
	//  конвертируем в HSV 

	// разбиваем на каналы
	//cvCvtPixToPlane(hsv, h_plane, s_plane, v_plane, 0);

	// создаём изображения
	//IplImage* hsv2 = cvCreateImage(cvGetSize(img2), 8, 3);
	//IplImage* h_plane = cvCreateImage(cvGetSize(src), 8, 1);
	//IplImage* s_plane = cvCreateImage(cvGetSize(src), 8, 1);
	//IplImage* v_plane = cvCreateImage(cvGetSize(src), 8, 1);
	//  конвертируем в HSV 


	//cvSmooth(img1, img1, CV_BLUR, 3, 3);
	//cvSmooth(img2, img2, CV_BLUR, 3, 3);


	/* float kernel[9];
	kernel[0] = 0.1;
	kernel[1] = 0.1;
	kernel[2] = 0.1;

	kernel[3] = 0.1;
	kernel[4] = 0.1;
	kernel[5] = 0.1;

	kernel[6] = 0.1;
	kernel[7] = 0.1;
	kernel[8] = 0.1;

	// матрица
	CvMat kernel_matrix = cvMat(3, 3, CV_32FC1, kernel);

	// накладываем фильтр
	cvFilter2D(img1, img1, &kernel_matrix, cvPoint(-1, -1));
	cvFilter2D(img2, img2, &kernel_matrix, cvPoint(-1, -1)); */

	//cvCvtColor(img1, hsv1, CV_BGR2HSV);
	//cvCvtColor(img2, hsv2, CV_BGR2HSV);

	// покажем изображения
	//cvNamedWindow("image2");
	//cvShowImage("image2", img2);

	// создаём картинку для хранения разницы
	if (diff==0)
	  diff = cvCloneImage(img1);

	//diff = cvCloneImage(img2);

	//sub = cvCloneImage(img1);

	cvZero(diff);

	fimg1 = cvCloneImage(img1);
	fimg2 = cvCloneImage(img2);


	for (int i = 0; i < ThreadsQuantity; i++)
	{
		pool.runAsync(&CheckSimil, i);
	}
	
	while (!pool.isEnd())
		cvWaitKey(1);

	//cvWaitKey(10);
	/*CheckSimil(0);
	CheckSimil(1);
	CheckSimil(2);
	CheckSimil(3);*/

	// пробегаемся по всем пикселям изображения
	/*for (int y = 0; y<diff->height; y++) {
		//uchar* ptr1 = (uchar*)(hsv1->imageData + y * hsv1->widthStep);
		//uchar* ptr2 = (uchar*)(hsv2->imageData + y * hsv2->widthStep);
		//uchar* ptr = (uchar*)(diff->imageData + y * diff->widthStep);
		uchar* ptr1 = (uchar*)(img1->imageData + y * img1->widthStep);
		uchar* ptr2 = (uchar*)(img2->imageData + y * img2->widthStep);
		uchar* ptr = (uchar*)(diff->imageData + y * diff->widthStep);
		for (int x = 0; x<diff->width; x++) {
			// 3 канала:
			if ((abs(ptr1[3 * x] - ptr2[3 * x]) > 70) || (abs(ptr1[3 * x + 1] - ptr2[3 * x + 1]) > 70) || (abs(ptr1[3 * x + 2] - ptr2[3 * x + 2]) > 70)) //&& (((abs(ptr1[3 * x] - ptr2[3 * x])) + (abs(ptr1[3 * x + 1] - ptr2[3 * x + 1])) + (abs(ptr1[3 * x + 2] - ptr2[3 * x + 2]))) > 100))
			//if (((abs(ptr1[3 * x] - ptr2[3 * x])) + (abs(ptr1[3 * x + 1] - ptr2[3 * x + 1])) + (abs(ptr1[3 * x + 2] - ptr2[3 * x + 2]))) > 100)
			{
				ptr[3 * x] = 127;
				ptr[3 * x + 1] = 127;	//Мой вариант
				ptr[3 * x + 2] = 127;
			}

			// B
			//ptr[3 * x] = ptr1[3 * x] + ptr2[3 * x] - 2 * min(ptr1[3 * x], ptr2[3 * x]);
			// G
			//ptr[3 * x + 1] = ptr1[3 * x + 1] + ptr2[3 * x + 1] - 2 * min(ptr1[3 * x + 1], ptr2[3 * x + 1]);
			// R
			//ptr[3 * x + 2] = ptr1[3 * x + 2] + ptr2[3 * x + 2] - 2 * min(ptr1[3 * x + 2], ptr2[3 * x + 2]);
		}
	}*/

	//cvSub(img2, img1, diff);

	IplImage* gray = 0;
	IplImage* bin = 0;

    gray = cvCreateImage(cvGetSize(diff), IPL_DEPTH_8U, 1);
	bin = cvCreateImage(cvGetSize(diff), IPL_DEPTH_8U, 1);

	//cvSmooth(diff, diff, CV_BLUR, 3, 3);

	cvCvtColor(diff, gray, CV_RGB2GRAY);

	//cvNamedWindow("gray");
	//cvShowImage("gray", gray);

	cvInRangeS(gray, cvScalar(40), cvScalar(150), bin); // atoi(argv[2])

	//cvNamedWindow("bin");
	//cvShowImage("bin", bin);

	CvMemStorage* storage = cvCreateMemStorage(0);
	CvSeq* contours = 0;
	

	vector<Point2f> tmp;

	int contoursCont = cvFindContours(bin, storage, &contours, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));

	for (CvSeq* seq0 = contours; seq0 != 0; seq0 = seq0->h_next){
		double area = fabs(cvContourArea(seq0));
		if ((area > 15000))									//МОЖНО ЗАДАТЬ ТРЕБУЕМУЮ ПЛОЩАДЬ КОНТУРА
		{
			//rect = cvBoundingRect(seq0, 1);
			//cvRect(diff,rect.x,rect.y,rect.width,rect.height);
			CvPoint2D32f center;
			float radius = 0;
			// находим параметры окружности
			cvMinEnclosingCircle(seq0, &center, &radius);
			//printf("Nayden obj s koord %d %d\n", int(center.x),int(center.y));

			Point2f point;
			int x = int(center.x);
			int y = int(center.y);
			point = Point2f((float)x, (float)y);
		    tmp.push_back(point);


			cvCircle(diff, cvPointFrom32f(center), radius, CV_RGB(255, 0, 0), 1, 8);
			//box = cvMinAreaRect2(seq0);
			//cvRect(box.center.x - box.size.width/2, box.center.y - box.size.height/2, box.size.width, box.size.height);
			cvDrawContours(diff, seq0, CV_RGB(255, 216, 0), CV_RGB(0, 0, 250), 0, 1, 8); // рисуем контур
		}

	}
	
	//for (int i = 0; i < tmp.size();i++)
	//{
	//	printf("Pt %d: %d %d\n", i, int(tmp[i].x), int(tmp[i].y));
	//}
	cvNamedWindow("contours");
	cvShowImage("contours", diff);


	cvReleaseMemStorage(&storage);
	// освобождаем ресурсы
	//cvReleaseImage(&img1);
	//cvReleaseImage(&img2);
	//cvReleaseImage(&diff);
	//cvReleaseImage(&fimg1);
	//cvReleaseImage(&fimg2);
	cvReleaseImage(&bin);
	cvReleaseImage(&gray);

	return tmp;

	//cvWaitKey(0);   !!!
	//cvNamedWindow("contours", CV_WINDOW_AUTOSIZE);

	//cvShowImage("contours", diff);
	// вычитаем 
	//cvSub(img2, img1, sub);
	/* int radius = 1;
	int iterations = 1;

	IplImage* erode = 0;
	IplImage* dilate = 0;

	erode = cvCloneImage(diff);
	dilate = cvCloneImage(diff);

	cvNamedWindow("erode", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("dilate", CV_WINDOW_AUTOSIZE);

	IplConvKernel* Kern = cvCreateStructuringElementEx(radius * 2 + 2, radius * 2 + 2, radius, radius, CV_SHAPE_ELLIPSE);

	// выполняем преобразования
	cvErode(diff, erode, Kern, iterations);
	cvDilate(diff, dilate, Kern, iterations);

	cvShowImage("erode", erode);
	cvShowImage("dilate", dilate); */

	// выводим результат
	//cvNamedWindow("diff");
	//cvShowImage("diff", diff);

	//cvNamedWindow("sub");
	//cvShowImage("sub", sub);

	// ждём нажатия клавиши
	//cvWaitKey(0);


	//cvDestroyWindow("contours");
	//cvReleaseImage(&sub);
	//cvReleaseImage(&erode);
	//cvReleaseImage(&dilate);
}
int main(int argc, char* argv[])
{

	
	//cap.open(0);
	CvCapture* capture = cvCreateCameraCapture(CV_CAP_ANY); //cvCaptureFromCAM( 0 );
	assert(capture);

	// дефолтные названия картинок для обработки
	//char file1[] = "Image0.jpg";
	//char file2[] = "Image1.jpg";

	// имя картинки задаётся первым параметром
	//char* filename1 = argc >= 2 ? argv[1] : "Image0.jpg";
	// получаем картинку
	//IplImage* img1 = cvLoadImage(filename1);

	//cvNamedWindow("image1");
	//cvShowImage("image1", img1);

	//cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, 640);//1280); 
	//cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, 480);//960); 

	// узнаем ширину и высоту кадра
	//double width = cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH);
	//double height = cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT);
	//printf("[i] %.0f x %.0f\n", width, height);

	IplImage* frame = 0;

	cvNamedWindow("capture", CV_WINDOW_AUTOSIZE);

	printf("[i] press Enter for capture image and Esc for quit!\n\n");

	int counter = 0;
	char filename[512];
	IplImage* image1 = 0;
	IplImage* prevImage = 0;
	IplImage* grey = 0;
	IplImage* prevGrey = 0;
	vector<Point2f> points1;
	int lol = 0;
	frame = cvQueryFrame(capture);
	grey = cvCreateImage(cvGetSize(frame), IPL_DEPTH_8U, 1);
	prevGrey = cvCreateImage(cvGetSize(frame), IPL_DEPTH_8U, 1);


	while (true)
	{
		// получаем кадр
		float fTimeStart;

		if (counter % 50 == 1)
			fTimeStart = omp_get_wtime();
		
		
		prevImage = cvCloneImage(frame);
		cvCvtColor(prevImage, prevGrey, CV_RGB2GRAY);

		frame = cvQueryFrame(capture);

		cvCvtColor(frame, grey, CV_RGB2GRAY);

		// показываем
		cvShowImage("capture", frame);

		char c = cvWaitKey(1);
		if (c == 27) { // нажата ESC
			break;
		}
		else if (c == 13) 
		{ // Enter
			// сохраняем кадр в файл
			if (counter == 0)
			{
				sprintf(filename, "ImageNew%d.jpg", counter);
				printf("Ishodnoe izobr sohraneno\n");
				cvSaveImage(filename, frame);
				counter++;
				image1 = cvCloneImage(frame);
				cvNamedWindow("image1");
				//namedWindow("LK Demo", 1);
				//image = Mat(image1, true);
				//imshow("LK Demo",image);
				cvShowImage("image1", image1);
			}
			//printf("[i] capture... %s\n", filename);
			//cvSaveImage(filename, frame);
			//counter++;
			//if (counter == 1)
			//{
				//image1 = cvCloneImage(frame); 
				//cvNamedWindow("image1");
				//cvShowImage("image1", image1);
			//}
			/*else if (counter > 0)
			{
				printf("Novoe izobr № %d\n",counter);
				showContur(image1, frame);
				//printf("Hello");
				counter++;
			}*/
		}
		if (counter > 0)
		{
			if ((prevImage != 0) && (!points1.empty()))			
				LucasKanade(Mat(prevGrey, true), Mat(grey, true), points1);
			points1= showContur(image1, frame);
			if (counter % 50 == 0)
			{
				float fTimeEnd = omp_get_wtime();
				printf("FPS %3.2f\n", 50.0 / (fTimeEnd - fTimeStart));
			}
			counter++;
		}
		//cvReleaseImage(&frame);
		cvReleaseImage(&prevImage);

		cvReleaseImage(&fimg1);
		cvReleaseImage(&fimg2);

		//cvReleaseImage(&grey);
		
	}


	// освобождаем ресурсы
	//cvReleaseCapture(&capture);
	//cvDestroyWindow("capture");
	//cvReleaseImage(&image1);
	//cvReleaseImage(&frame);

	// удаляем окна
	cvDestroyAllWindows();
	return 0;
}