#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

Mat calcGrayHist(const Mat& img);
Mat getGrayHistImage(const Mat& hist);

void HistPlay();
void histogram_stretching();
void histogram_equalization();
void camera_in();
void video_in();
void camera_in_video_out();
void picAdd();
void picAdd_2();
void picAddWeghited();

int main(void)
{
	//HistPlay();
	//histogram_stretching();
	//video_in();

	//histogram_equalization();

	//picAdd_2();

	picAddWeghited();

	return 0;
}

// ¿µ»óÀÇ µ¡¼À¿¬»ê add()ÇÔ¼ö»ç¿ë
void picAdd()
{
	Mat cam = imread("camera.bmp", IMREAD_GRAYSCALE);
	Mat aer = imread("aero2.bmp", IMREAD_GRAYSCALE);

	Mat def;
	add(cam, aer, def);

	imshow("µ¡¼À °á°ú", def);
	waitKey();
}


// ¿µ»óÀÇ µ¡¼À¿¬»ê2
void picAdd_2()
{
	Mat cam = imread("camera.bmp", IMREAD_GRAYSCALE);
	Mat aer = imread("aero2.bmp", IMREAD_GRAYSCALE);

	Mat def;
	def = cam + aer;

	imshow("µ¡¼À °á°ú", def);
	waitKey();
}

void picAddWeghited()
{
	Mat cam = imread("camera.bmp", IMREAD_GRAYSCALE);
	Mat aer = imread("aero2.bmp", IMREAD_GRAYSCALE);

	Mat def;

	addWeighted(cam, 0.5, aer, 0.5,0, def);

	imshow("µ¡¼À °á°ú", def);
	waitKey();
}


void histogram_stretching()
{
	Mat src = imread("hawkes.bmp", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}

	double gmin, gmax;
	minMaxLoc(src, &gmin, &gmax);

	Mat dst = (src - gmin) * 255 / (gmax - gmin);

	imshow("src", src);
	imshow("srcHist", getGrayHistImage(calcGrayHist(src)));
	
	imshow("dst", dst);
	imshow("dstHist", getGrayHistImage(calcGrayHist(dst)));

	waitKey();
	destroyAllWindows();

}

void histogram_equalization()
{
	Mat src = imread("hawkes.bmp", IMREAD_GRAYSCALE);

	if (src.empty())
	{
		cerr << "Image load failed" << endl;
		return;
	}

	Mat dst;

	// OpenCV¿¡¼­ Á¦°øÇÏ´Â È÷½ºÅä±×·¥ ÆòÈ°È­ ÇÔ¼ö
	equalizeHist(src, dst);

	imshow("src", src);
	imshow("srcHist", getGrayHistImage(calcGrayHist(src)));
	
	imshow("dst", dst);
	imshow("dstHist", getGrayHistImage(calcGrayHist(dst)));

	waitKey();
	destroyAllWindows();
}

void HistPlay()
{
	Mat src = imread("lenna.bmp", IMREAD_GRAYSCALE);
	Mat hist = calcGrayHist(src);
	Mat hist_img = getGrayHistImage(hist);

	imshow("src", src);
	imshow("srcHist", hist_img);

	waitKey();
}

Mat calcGrayHist(const Mat& img)
{
	CV_Assert(img.type() == CV_8UC1);

	Mat hist;
	int channels[] = { 0 };
	int dims = 1;
	const int histSize[] = { 256 };
	float graylevel[] = { 0,256 };
	const float* ranges[] = { graylevel };

	calcHist(&img, 1,channels, noArray(), hist, dims, histSize, ranges);

	return hist;
}

Mat getGrayHistImage(const Mat& hist)
{
	CV_Assert(hist.type() == CV_32FC1);
	CV_Assert(hist.size() == Size(1, 256));

	double histMax;
	minMaxLoc(hist, 0, &histMax);

	Mat imgHist(100, 256, CV_8UC1, Scalar(256));
	for (int i = 0; i < 256; i++)
	{
		line(imgHist, Point(i, 100),
			Point(i, 100 - cvRound(hist.at<float>(i, 0) * 100 / histMax)), Scalar(0));
	}

	return imgHist;
}


void camera_in()
{
	VideoCapture cap(0);

	if (!cap.isOpened()) {
		cerr << "Camera open failed!" << endl;
		return;
	}

	cout << "Frame width: " << cvRound(cap.get(CAP_PROP_FRAME_WIDTH)) << endl;
	cout << "Frame height: " << cvRound(cap.get(CAP_PROP_FRAME_HEIGHT)) << endl;

	Mat frame, inversed;
	while (true) {
		cap >> frame;
		if (frame.empty())
			break;

		inversed = ~frame;

		imshow("frame", frame);
		imshow("inversed", inversed);

		if (waitKey(10) == 27) // ESC key
			break;
	}

	destroyAllWindows();
}

void video_in()
{
	VideoCapture cap("stopwatch.avi");

	if (!cap.isOpened()) {
		cerr << "Video open failed!" << endl;
		return;
	}

	cout << "Frame width: " << cvRound(cap.get(CAP_PROP_FRAME_WIDTH)) << endl;
	cout << "Frame height: " << cvRound(cap.get(CAP_PROP_FRAME_HEIGHT)) << endl;
	cout << "Frame count: " << cvRound(cap.get(CAP_PROP_FRAME_COUNT)) << endl;

	double fps = cap.get(CAP_PROP_FPS);
	cout << "FPS: " << fps << endl;

	int delay = cvRound(1000 / fps);

	Mat frame, inversed;
	while (true) {
		cap >> frame;
		if (frame.empty())
			break;

		inversed = ~frame;

		imshow("frame", frame);
		imshow("inversed", inversed);

		if (waitKey(delay) == 27) // ESC key
			break;
	}

	destroyAllWindows();
}

void camera_in_video_out()
{
	VideoCapture cap(0);

	if (!cap.isOpened()) {
		cerr << "Camera open failed!" << endl;
		return;
	}

	int w = cvRound(cap.get(CAP_PROP_FRAME_WIDTH));
	int h = cvRound(cap.get(CAP_PROP_FRAME_HEIGHT));
	double fps = cap.get(CAP_PROP_FPS);

	int fourcc = VideoWriter::fourcc('D', 'I', 'V', 'X');
	int delay = cvRound(1000 / fps);

	VideoWriter outputVideo("output.avi", fourcc, fps, Size(w, h));

	if (!outputVideo.isOpened()) {
		cout << "File open failed!" << endl;
		return;
	}

	Mat frame, inversed;
	while (true) {
		cap >> frame;
		if (frame.empty())
			break;

		inversed = ~frame;
		outputVideo << inversed;

		imshow("frame", frame);
		imshow("inversed", inversed);

		if (waitKey(delay) == 27)
			break;
	}

	destroyAllWindows();
}
