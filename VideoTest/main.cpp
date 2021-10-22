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
void picSub();
void picDiff();
void picLogical();

void filter_embossing();

void blurring_mean();
void blurring_gaussian();

void unsharped_mask();
void filter_bilateral();
void filter_median();

void output();
void affine_transform();
void affine_translation();

void affine_shear();
void affine_scale();


int main(void)
{
	//HistPlay();					// 히스토그램 실행
	//histogram_stretching();		// 히스토그램 스트레칭
	//video_in();					// 노트북 비디오 출력

	//histogram_equalization();		// 히스토그램 평활화

	//picAdd_2();

	//picAddWeghited();	// 가중치 덧셈연산

	//picSub();

	//picDiff();
	//picLogical();	// 논리적 연산

	//filter_embossing();

	//blurring_mean();		// 평균값 블러링
	//blurring_gaussian();	// 가우시안 블러링


	//unsharped_mask(); // 언샤프 마스크

	//filter_bilateral();	//양방향 필터

	//filter_median(); // 미디언 필터(중간값 필터)

	//output();	// 출력 확인

	//affine_transform(); // 어파인 변환 = 평행이동시키거나 회전, 크기 변환 등을 통해 만들 수 있는 변환을 통칭
	//affine_translation(); // 이동 변환

	//affine_shear(); // 전단 변환

	affine_scale();		// 크기 변환

	return 0;
}

// 영상의 덧셈연산 add()함수사용
void picAdd()
{
	Mat cam = imread("camera.bmp", IMREAD_GRAYSCALE);
	Mat aer = imread("aero2.bmp", IMREAD_GRAYSCALE);

	Mat def;
	add(cam, aer, def);

	imshow("덧셈 결과", def);
	waitKey();
}


// 영상의 덧셈연산2
void picAdd_2()
{
	Mat cam = imread("camera.bmp", IMREAD_GRAYSCALE);
	Mat aer = imread("aero2.bmp", IMREAD_GRAYSCALE);

	Mat def;
	def = cam + aer;

	imshow("덧셈 결과", def);
	waitKey();
}


// 평균연산 addWeighted()함수 사용하여서 가중치를 설정하고 포화되는 픽셀이 없도록 연산
void picAddWeghited()
{
	Mat cam = imread("camera.bmp", IMREAD_GRAYSCALE);
	Mat aer = imread("aero2.bmp", IMREAD_GRAYSCALE);

	Mat def;

	addWeighted(cam, 0.5, aer, 0.5,0, def);

	imshow("덧셈 결과", def);
	waitKey();
}

void picSub()
{

	Mat pic1 = imread("lenna.bmp", IMREAD_GRAYSCALE);
	Mat pic2 = imread("hole2.bmp", IMREAD_GRAYSCALE);

	Mat def1;
	Mat def2;

	subtract(pic1, pic2, def1);
	subtract(pic2, pic1, def2);
	imshow("뺄셈 결과1", def1);
	imshow("뺄셈 결과2", def2);

	waitKey();
}

void picDiff()
{
	Mat pic1 = imread("lenna.bmp", IMREAD_GRAYSCALE);
	Mat pic2 = imread("square.bmp", IMREAD_GRAYSCALE);

	Mat def;
	
	absdiff(pic1, pic2, def);
	imshow("결과", def);
	waitKey();

}

void picLogical()
{
	Mat pic1 = imread("lenna.bmp", IMREAD_GRAYSCALE);
	Mat pic2 = imread("square.bmp", IMREAD_GRAYSCALE);

	Mat def1,def2,def3,def4;

	bitwise_and(pic1, pic2, def1);
	bitwise_or(pic1, pic2, def2);
	bitwise_xor(pic1, pic2, def3);
	bitwise_not(pic1,  def4);

	imshow("def1", def1);
	imshow("def2", def2);
	imshow("def3", def3);
	imshow("def4", def4);

	waitKey();

}

void filter_embossing()
{
	Mat pic = imread("lenna.bmp", IMREAD_GRAYSCALE);

	float data1[] = { -1,-1,0,-1,0,1,0,1,1 };
	Mat emboss1(3, 3, CV_32FC1, data1);

	float data2[] = { -2,-2,0,-2,0,2,0,2,2 };
	Mat emboss2(3, 3, CV_32FC1, data2);

	Mat dst1,dst2;
	filter2D(pic, dst1, -1, emboss1, Point(-1, -1), 128);
	filter2D(pic, dst2, -1, emboss2, Point(-1, -1), 128);

	imshow("dst1", dst1);
	imshow("dst2", dst2);

	waitKey();
}

//평균값 블러링
void blurring_mean()
{
	Mat pic = imread("IU2.bmp", IMREAD_GRAYSCALE);

	imshow("pic", pic);

	Mat dst;
	for (int ksize = 3; ksize <= 30; ksize++) {
		blur(pic, dst, Size(ksize, ksize));

		String desc = format("Mean: %d x %d", ksize, ksize);
		putText(dst, desc, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255), 1, LINE_AA);

		imshow("dst", dst);
		waitKey();
	}
	destroyAllWindows();
}

void blurring_gaussian()
{
	Mat pic = imread("rose.bmp", IMREAD_GRAYSCALE);
	
	imshow("pic", pic);

	Mat def;
	for (int sigma = 1; sigma <= 20; sigma++)
	{
		GaussianBlur(pic, def, Size(), (double)sigma);

		String text = format("sigma = %d", sigma);
		putText(def, text, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255),1, LINE_AA);

		imshow("def", def);
		waitKey();
	}
	destroyAllWindows();
}

void unsharped_mask()
{
	Mat pic = imread("sa.bmp", IMREAD_COLOR);

	imshow("pic", pic);

	for (int sigma = 1; sigma <= 20; sigma++)
	{
		Mat blurred;
		GaussianBlur(pic, blurred, Size(), sigma);

		float alpha = 1.f;
		Mat dst = (1 + alpha) * pic - alpha * blurred;

		String desc = format("sigma: %d", sigma);
		putText(dst, desc, Point(10, 30), FONT_HERSHEY_PLAIN, 1.0,
			Scalar(255), 1, LINE_AA);
		
		imshow("blurred", blurred);
		imshow("dst", dst);
		waitKey();
	}
	destroyAllWindows();
}

void filter_bilateral()
{
	Mat pic = imread("IU2.bmp", IMREAD_GRAYSCALE);

	if (pic.empty())
	{
		cerr << "Image load failed!" << endl;
		return;
	}

	Mat noise(pic.size(), CV_32SC1);
	randn(noise, 0, 5);
	add(pic, noise, pic, Mat(), CV_8U);

	Mat dst1;
	GaussianBlur(pic, dst1, Size(), 5);

	Mat dst2;
	bilateralFilter(pic, dst2, -1, 10, 5);

	imshow("pic",pic);
	imshow("dst1", dst1);
	imshow("dst2", dst2);

	waitKey();
	
}

void filter_median()
{
	Mat pic = imread("IU2.bmp", IMREAD_COLOR);

	

	int num = (int)(pic.total() * 0.1);
	// 소금&후추 잡음 설정
	for (int i = 0; i < num; i++)
	{
		int x = rand() % pic.cols;
		int y = rand() % pic.rows;
		pic.at<uchar>(y, x) = (i % 2) * 255;
	}
	imshow("소금&후추 잡음", pic);
	Mat dst1;
	GaussianBlur(pic, dst1, Size(), 1);

	// 가우시안 필터 사용으로 잡음 제거 시도
	Mat dst2;
	medianBlur(pic, dst2, 3);

	// 미디언 필터 사용으로 잡음 제거 시도
	Mat dst3;
	medianBlur(dst1, dst3, 3);

	imshow("가우시안", dst1);
	imshow("미디언", dst2);
	imshow("가우시안->미디언", dst3);


	waitKey();
}

void output()
{
	Mat pic = imread("sa.bmp", IMREAD_GRAYSCALE);

	imshow("IU", pic);

	waitKey();
}

void affine_transform()
{
	Mat pic = imread("sa.bmp");

	Point2f picPts[3], dstPts[3];
	picPts[0] = Point2f(0, 0);
	picPts[1] = Point2f(pic.cols - 1, 0);
	picPts[2] = Point2f(pic.cols - 1, pic.rows - 1);
	dstPts[0] = Point2f(50, 50);
	dstPts[1] = Point2f(pic.cols - 100, 100);
	dstPts[2] = Point2f(pic.cols - 50, pic.rows - 50);

	Mat M = getAffineTransform(picPts, dstPts);

	Mat dst;
	warpAffine(pic, dst, M, Size());

	imshow("pic", pic);
	imshow("dst", dst);

	waitKey();
	destroyAllWindows();
}

void affine_translation()
{
	Mat pic = imread("IU2.bmp");

	Mat M = Mat_<double>({ 2,3 }, {1,0,150,0,1,100});

	Mat dst;
	warpAffine(pic, dst, M, Size());

	imshow("pic", pic);
	imshow("dst", dst);

	waitKey();


}

void affine_shear()
{
	Mat pic = imread("IU2.bmp");

	double mx = 0.3;
	Mat M = Mat_<double>({ 2,3 }, { 1,mx,0,0,1,0 });

	Mat dst;
	warpAffine(pic, dst, M, Size(cvRound(pic.cols + pic.rows * mx), pic.rows));

	imshow("pic", pic);
	imshow("dst", dst);

	waitKey();
	destroyAllWindows();
}

void affine_scale()
{
	Mat pic = imread("lenna_Color.bmp");

	Mat dst1, dst2, dst3, dst4;

	resize(pic, dst1, Size(), 4, 4, INTER_NEAREST);
	resize(pic, dst2, Size(1920, 1280));
	resize(pic, dst3, Size(1920, 1280),0,0,INTER_CUBIC);
	resize(pic, dst4, Size(1920, 1280),0,0,INTER_LANCZOS4);

	imshow("pic", pic);
	imshow("dst1", dst1(Rect(800,500,400,400)));
	imshow("dst2", dst2(Rect(800, 500, 400, 400)));
	imshow("dst3", dst3(Rect(800, 500, 400, 400)));
	imshow("dst4", dst4(Rect(800, 500, 400, 400)));

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

	// OpenCV에서 제공하는 히스토그램 평활화 함수
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
