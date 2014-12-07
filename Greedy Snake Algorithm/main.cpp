//#include "stdafx.h"

#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv/cv.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>



using namespace std;
using namespace cv;

/* Defines */
#define STATIC_IMAGE_MODE		/* For tracking an object on video comment this out */
#define SAMPLE_PERIOD 10		/* number of mouse moves in order to get a contour point */
#define FEEDBACK_CONST 5 
#define curvatureThreshold 5	
#define magnitudeThreshold 3		/* Ditto */
#define pointsMovedThreshold 10		/* min points needed to move (in this iter) before we move on to next iter */
#define MAX_POTENTIAL_VALUE 120
#define MAX_TICK_COUNT 1000000
/* End Defines*/

/* Global Variables */
#ifndef STATIC_IMAGE_MODE
	VideoCapture cap("C:\\Users\\USER\\Documents\\GitHub\\GreedySnake\\Debug\\IMG_0543.mov");
#endif
VideoWriter outputVideo;
double E_min;
Mat blankCanvas;
Mat canvas;
Mat canvasGrey;
Mat cannyUnblurred;
Mat cannyOutput;
bool drawingNow; 
bool doneDrawing;
int sampleSoFar;
vector<Point> mouseContour;
vector<double> alpha;
vector<double> beta;
vector<double> gamma;
vector<double> fdb;		/* Feedback constants */


bool stopGoing = false;
int useAttractable = 1;
int AMOUNT_OF_BLUR = 5;			/* Increases size of the edge */
int CANNY_THRESHOLD = 100;
int ALPHA = 10;						/* Econt */
int BETA  = 10;						/* Ecurv */
int GAMMA = 80;					/* Eimage*/
int NEIGHBOURHOOD_SIZE = 3;		/* numNeighbours(including self) = (2*NEIGHBOURHOOD_SIZE + 1)^2 */

double AVE_DIST_CHANGE_THRESHOLD = 0.0001;
int distPercentage = 1;

/* End Global Variables*/

/* Higher Level Functions */
void preprocessImage(void);
void getPointsWithMouse(void);
void mouseDrawCallback(int event, int x, int y, int flags, void* userdata);
bool greedyAttractableSnake();
/* End Higher Level Functions */

/* Helping functions */
double Econt(int i, Point j, double aveDist, double maxMove);
double Ecurv(int i, Point j, double maxCurv);
double Eimage(int i, Point j);
double gradPfield(int i);
Point2d normalDirection(int i);
double projNormal(Point2d dir, Point2d normDir);
double maxPfield(int i);
double avePfield(int i);
double aveDistBtwnPts();		// sum_0_to_n-1 { |V_i - V_i-1| / n }
double maxMovingDist(int i, double aveDist);	// max_j { aveDistBtwnPts - |V_j - V_i-1| }
double maxCurvature(int i); // max_j { |V_i-1 - 2*V_j + V_i+1| }
/* End Happy functions*/

/* SOME THRESHOLD/CHANGING CALLBACKS */

void avePointsThreshCallback(int, void*);
void cannyThreshCallback(int, void*);
void blurCallback(int, void*);
void mouseStopCallback(int event, int x, int y, int flags, void* userdata);

/* END SOME THRESHOLD/CHANGING CALLBACKS */

int main()
{
#ifndef STATIC_IMAGE_MODE
		if (!cap.isOpened())
		{
			return -1;
		}

		Size S = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),    // Acquire input size
			(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));
		int ex = static_cast<int>(cap.get(CV_CAP_PROP_FOURCC));
		outputVideo.open("C:\\Users\\USER\\Documents\\GitHub\\GreedySnake\\Debug\\OutputVideo.avi", CV_FOURCC('P', 'I', 'M', '1'), 20, S, true);
		if (!outputVideo.isOpened())
		{
			cout << "Could not open the output video for write: OutputVideo.avi" << endl;
			return -1;
		}
#endif

		

	preprocessImage(); // Detect edges with Canny. Maybe do some other things later on?

	char* cannyWindow = "KNOBS";
	namedWindow(cannyWindow, CV_WINDOW_NORMAL);
	createTrackbar("Use Attractable Snake:", "KNOBS", &useAttractable, 1, NULL);
	createTrackbar("Blur:", "KNOBS", &AMOUNT_OF_BLUR, 20, blurCallback);
	createTrackbar("Canny Thresh:", "KNOBS", &CANNY_THRESHOLD, 300, cannyThreshCallback);
	createTrackbar("AvgPtDist Thresh:", "KNOBS", &distPercentage, 100, avePointsThreshCallback);
	createTrackbar("Neighbourhood size:", "KNOBS", &NEIGHBOURHOOD_SIZE, 20, NULL);
	createTrackbar("ALPHA:", "KNOBS", &GAMMA, 100, NULL);
	createTrackbar("BETA:", "KNOBS", &BETA, 100, NULL);
	createTrackbar("GAMMA:", "KNOBS", &GAMMA, 100, NULL);

	//char* cannyOutputWindow = "CANNY";
	//namedWindow(cannyOutputWindow, CV_WINDOW_AUTOSIZE);
	//imshow(cannyOutputWindow, cannyOutput);

	getPointsWithMouse();

	waitKey();
	setMouseCallback("DRAW!", mouseStopCallback, NULL); 
	cout << "START GREEDY ALGORITHM!\n";
	
	alpha.assign(mouseContour.size(), ALPHA);
	beta.assign(mouseContour.size(), BETA);
	gamma.assign(mouseContour.size(), GAMMA);
	fdb.assign(mouseContour.size(), FEEDBACK_CONST);
	//int numFramesSkipped = 0;
	if (!greedyAttractableSnake())
	{
		cout << "FAIL: GREEDY RETURNED FALSE!\n";
		return -1;
	}

#ifndef STATIC_IMAGE_MODE
	while (!canvas.empty())
	{
		if (!greedyAttractableSnake())
		{
			cout << "FAIL: GREEDY RETURNED FALSE!\n";
			break;
		}
		cout << "NEXT\n";	
		preprocessImage();
		//numFramesSkipped = 0;
		//else
		//{
		//	cap >> canvas;
		//	if (canvas.empty())
		//		break;
		//}
			
	}
#endif
	cout << "FINISHED GREEDY ALGORITHM!\n";
	
	/* This just stops the program from finishing */
	int foo;
	cin >> foo;

	return 0;
}

void preprocessImage(void)
{

#ifdef STATIC_IMAGE_MODE
	canvas = imread("C:\\Users\\USER\\Documents\\GitHub\\GreedySnake\\Debug\\rat.jpg");
	if (canvas.empty())
		return;
	Size frameSize(canvas.cols, canvas.rows);
	outputVideo.open("C:\\Users\\USER\\Documents\\GitHub\\GreedySnake\\Debug\\OutputVideo.avi", CV_FOURCC('P', 'I', 'M', '1'), 20, frameSize, true);
#else	
	// If we do video input, this needs to read a frame of the video
	cap >> canvas;
#endif
	blankCanvas = canvas.clone();
	// Convert image to grey and blur
	cvtColor(canvas, canvasGrey, CV_BGR2GRAY);
	blur(canvasGrey, canvasGrey, Size(3, 3));

	// Detect edges using canny
	Canny(canvasGrey, cannyUnblurred, CANNY_THRESHOLD, CANNY_THRESHOLD * 2, 3);

	// Blur canny for better effect with greedy algorithm (more area of effect! probably)
	blur(cannyUnblurred, cannyOutput, Size(AMOUNT_OF_BLUR, AMOUNT_OF_BLUR));
	//imshow("CANNY", cannyOutput);

	return;
}

void getPointsWithMouse()
{
	blankCanvas = canvas.clone();
	char* window = "DRAW!";
	namedWindow(window, CV_WINDOW_AUTOSIZE);
	imshow(window, canvas);

	
	/* Initialise global drawing variables */
	drawingNow = false;
	doneDrawing = false;
	sampleSoFar = 0;

	setMouseCallback("DRAW!", mouseDrawCallback, &mouseContour);
	
}

void mouseDrawCallback(int event, int x, int y, int flags, void* userdata)
{

	if (event == EVENT_LBUTTONDOWN)
	{
		mouseContour.clear();
		blankCanvas.copyTo(canvas);
		cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
		// Create Point At the position where mouse clicks
		mouseContour.push_back(Point(x, y));
		sampleSoFar = 0;
		drawingNow = true;
		doneDrawing = false;
	}
	//else if (event == EVENT_RBUTTONDOWN)
	//{
	//	cout << "Right button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
	//}
	//else if (event == EVENT_MBUTTONDOWN)
	//{
	//	cout << "Middle button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
	//}
	else if (event == EVENT_MOUSEMOVE)
	{
		if (drawingNow == true)
		{
			//cout << "Mouse move over the window - position (" << x << ", " << y << ")" << endl;
			if (sampleSoFar < SAMPLE_PERIOD)
			{
				sampleSoFar++;
				return;
			}
			else if (sampleSoFar == SAMPLE_PERIOD)
			{
				sampleSoFar = 1;
			}
			mouseContour.push_back(Point(x, y));

			int thickness = 2;
			int lineType = 8;
			int size = mouseContour.size() - 1;
			for (int i = 0; i < size; i++)
			{
				line(canvas, mouseContour.at(i), mouseContour.at(i + 1), Scalar(0, 200, 0), thickness, lineType);
			}
			imshow("DRAW!", canvas);
			outputVideo << canvas;
		}
	}
	else if (event == EVENT_LBUTTONUP)
	{
		if (mouseContour.size() > 0)
		{
			int thickness = 2;
			int lineType = 8;
			line(canvas, mouseContour.at(mouseContour.size() - 1), mouseContour.at(0), Scalar(0, 200, 0), thickness, lineType);
			imshow("DRAW!", canvas);
		}
			
		drawingNow = false;
		doneDrawing = true;


	}
}

void mouseStopCallback(int event, int x, int y, int flags, void* userdata)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		stopGoing = true;
	}
}

bool greedyAttractableSnake()
{
	int pointsMoved = 0;
	int numVertices = mouseContour.size();
	if (numVertices < 4) // Won't handle tiny contours 
	{
		return false;
	}
	E_min = 100000; /* TODO: Change this to something that actually represents max double value*/
	//imshow("CANNY", cannyOutput);
	auto t1 = std::chrono::high_resolution_clock::now();
	double aveDist;
	do {
		aveDist = aveDistBtwnPts();
		pointsMoved = 0;
		for (int k = 0; k < numVertices + 1; k++) 
		{
			int i = k%numVertices;
			E_min = 100000; /* TODO: Change this to something that actually represents max double value*/
			double E_j;
			Point j_min(-1, -1);
			if (NEIGHBOURHOOD_SIZE == 0)
			{
				break;
			}
			int numRows = NEIGHBOURHOOD_SIZE * 2 + 1;
			int numNeighbours = pow(numRows, 2);
			double maxMove = maxMovingDist(i, aveDist);
			double maxCurv = maxCurvature(i);
			Point2d normalDir = normalDirection(i);
			double feedbackTerm = 0;
			if (useAttractable)
			{
				feedbackTerm = fdb.at(i)   * gradPfield(i);
			}

			/* MOVE POINTS TO PLACES WHERE THEY ARE LESS ENERGETIC */
			for (int j = 0; j < numNeighbours; j++)
			{
				int x_j = (j % numRows) - 1 + mouseContour.at(i).x;
				int y_j = (j / numRows) - 1 + mouseContour.at(i).y;
				if (x_j >= canvas.cols || y_j >= canvas.rows || x_j <= 0 || y_j <= 0)
				{
					continue;
				}
				Point point_j(x_j, y_j);
				Point2d dir(x_j - mouseContour.at(i).x, y_j - mouseContour.at(i).y);
				E_j = ALPHA * Econt(i, point_j, aveDist, maxMove)  
					+ BETA  * Ecurv(i, point_j, maxCurv)
					+ GAMMA * Eimage(i, point_j);
				if (useAttractable)
				{
					E_j -= projNormal(dir, normalDir) * feedbackTerm;
				}
					
				/* TODO: ALSO MINUS THE WEIRD FEEDBACK TERM TO THE E_j */
				if (E_j < E_min)
				{
					E_min = E_j;
					j_min.x = point_j.x;
					j_min.y = point_j.y;
				}
			}
			if (mouseContour.at(i) != j_min && j_min.x > 0 && j_min.y > 0)
			{
				assert(j_min.x < canvas.cols && j_min.y < canvas.rows && j_min.x > 0 && j_min.y > 0);
				mouseContour.at(i).x = j_min.x;
				mouseContour.at(i).y = j_min.y;
  				pointsMoved++;
			}
			/* End MOVE POINTS TO PLACES WHERE THEY ARE LESS ENERGETIC */

			/* IF GETS TOO CURVY, SET Ecurv TO 0 for point i (by setting beta(i) to 0) */
			/*vector<double> curvature;

			for (int i = 0; i < numVertices; i++)
			{
				double x_i = mouseContour.at(i).x;
				double y_i = mouseContour.at(i).y;
				double x_prev, y_prev, x_next, y_next;
				if (i == 0)
				{
					x_prev = mouseContour.at(numVertices - 1).x;
					y_prev = mouseContour.at(numVertices - 1).y;
				}
				else
				{
					x_prev = mouseContour.at(i - 1).x;
					y_prev = mouseContour.at(i - 1).y;
				}
				if (i == numVertices - 1)
				{
					x_next = mouseContour.at(0).x;
					y_next = mouseContour.at(0).y;
				}
				else
				{
					x_next = mouseContour.at(i + 1).x;
					y_next = mouseContour.at(i + 1).y;
				}
				Point u_i, u_next;
				u_i.x = x_i - x_prev;
				u_i.y = y_i - y_prev;
				u_next.x = x_next - x_i;
				u_next.y = y_next - y_i;

				double abs_u_i = hypot(u_i.x, u_i.y);
				double abs_u_next = hypot(u_next.x, u_next.y);
				
				curvature.push_back(pow(((u_i.x / abs_u_i) - (u_next.x / abs_u_next)), 2)
								  + pow(((u_i.y / abs_u_i) - (u_next.y / abs_u_next)), 2));
			}

			for (int i = 0; i < numVertices; i++)
			{
				double c_prev = i>0 ? curvature.at(i - 1) : curvature.at(numVertices - 1);
				double c_next = i < numVertices - 1 ? curvature.at(i + 1) : curvature.at(0);
				double c_i = curvature.at(i);
				if (c_i > c_prev && c_i > c_next 
					&& c_i > curvatureThreshold 
					&& hypot(mouseContour.at(i).x, mouseContour.at(i).y) > magnitudeThreshold)
				{
					beta.at(i) = 0;
				}
			}*/
			/* End IF GETS TOO CURVY, SET Ecurv TO 0 for point i */

		}

		waitKey(1);
		/* DRAW THE CONTOUR */
		blankCanvas.copyTo(canvas);
		int thickness = 2;
		int lineType = 8;
		int size = mouseContour.size() - 1;
		for (int i = 0; i < size; i++)
		{
			line(canvas, mouseContour.at(i), mouseContour.at(i + 1), Scalar(0, 200, 0), thickness, lineType);
		}
		if (mouseContour.size() > 1)
		{
			line(canvas, mouseContour.at(mouseContour.size() - 1), mouseContour.at(0), Scalar(0, 200, 0), thickness, lineType);
			imshow("DRAW!", canvas);
			outputVideo << canvas;
		}
		/* END DRAW THE CONTOUR */

#ifndef STATIC_IMAGE_MODE
		auto t2 = std::chrono::high_resolution_clock::now();
		auto timeDiff = t2 - t1;
		if (stopGoing)
		{
			stopGoing = false;
			break;
		}
		if (timeDiff.count() > MAX_TICK_COUNT)
			break;
#endif
		if (useAttractable && abs(aveDist - aveDistBtwnPts()) < AVE_DIST_CHANGE_THRESHOLD)
		{
			break;
		}
			
	} while (pointsMoved > pointsMovedThreshold); // as long as we're moving enough points, keep going!
												
	return true;
}

double Econt(int i, Point j, double aveDist, double maxMove)
{
	assert(maxMove >= 0 && i >= 0 && i < mouseContour.size() && aveDist >= 0);
	Point prevPt;
	if (i > 0)
	{
		prevPt.x = mouseContour.at(i - 1).x;
		prevPt.y = mouseContour.at(i - 1).y;
	}
	else
	{
		prevPt.x = mouseContour.at(mouseContour.size() - 1).x;
		prevPt.y = mouseContour.at(mouseContour.size() - 1).y;
	}

	return (aveDist - hypot(j.x - prevPt.x, j.y - prevPt.y)) / maxMove;
}
double Ecurv(int i, Point j, double maxCurv)
{
	assert(maxCurv >= 0 && i >= 0 && i < mouseContour.size());
	Point prevPt, nextPt;
	if (i > 0)
	{
		prevPt.x = mouseContour.at(i - 1).x;
		prevPt.y = mouseContour.at(i - 1).y;
	}
	else
	{
		prevPt.x = mouseContour.at(mouseContour.size() - 1).x;
		prevPt.y = mouseContour.at(mouseContour.size() - 1).y;
	}
	if (i < mouseContour.size() - 1)
	{
		nextPt.x = mouseContour.at(i + 1).x;
		nextPt.y = mouseContour.at(i + 1).y;
	}
	else
	{
		nextPt.x = mouseContour.at(0).x;
		nextPt.y = mouseContour.at(0).y;
	}

	return (pow(hypot(prevPt.x - 2*j.x + nextPt.x,
					  prevPt.y - 2*j.y + nextPt.y), 2) / maxCurv);
}
double Eimage(int i, Point j)
{
	//assert(j.x < canvas.cols && j.x > 0 && j.y < canvas.rows && j.y > 0);

	return (-1.0 * (double)cannyOutput.at<uchar>(j));
}

double gradPfield(int i)
{
	double Pmax = maxPfield(i);
	double Pavg = avePfield(i);
	return (Pmax - Pavg) / Pmax;
}

Point2d normalDirection(int i) /* TODO: Maybe do a better estimate of the normal? */
{
	assert(i >= 0 && i < mouseContour.size());
	int x1, y1, x2, y2;
	int last = mouseContour.size() - 1;
	if (i > 0)
	{
		x1 = mouseContour.at(i - 1).x - mouseContour.at(i).x;
		y1 = mouseContour.at(i - 1).y - mouseContour.at(i).y;
	}
	else
	{
		x1 = mouseContour.at(last).x - mouseContour.at(0).x;
		y1 = mouseContour.at(last).y - mouseContour.at(0).y;
	}
	if (i < last)
	{
		x2 = mouseContour.at(i + 1).x - mouseContour.at(i).x;
		y2 = mouseContour.at(i + 1).y - mouseContour.at(i).y;
	}
	else
	{
		x2 = mouseContour.at(0).x - mouseContour.at(last).x;
		y2 = mouseContour.at(0).y - mouseContour.at(last).y;

	}

	int newx = x1 + x2;
	int newy = y1 + y2;
	
	double len = hypot(newx, newy);

	Point2d ret;
	ret.x = newx / len;
	ret.y = newy / len;

	return ret;
}

double projNormal(Point2d dir, Point2d normDir) /* TODO: Figure out how this thing works! */
{
	/* TODO */
	double len = hypot(dir.x, dir.y);
	Point2d dirNormalised(dir.x / len, dir.y / len);
	return dirNormalised.x*normDir.x + dirNormalised.y*normDir.y;
}

double maxPfield(int i)
{
	return MAX_POTENTIAL_VALUE;
	//assert(i >= 0 && i < mouseContour.size());
	//int numRows = NEIGHBOURHOOD_SIZE * 2 + 1;
	//int numNeighbours = pow(numRows, 2);
	//double maxSoFar = (double)cannyOutput.at<uchar>(mouseContour.at(i));

	//for (int j = 0; j < numNeighbours; j++)
	//{
	//	int x_j = (j % numRows) - 1 + mouseContour.at(i).x;
	//	int y_j = (j / numRows) - 1 + mouseContour.at(i).y;
	//	if (x_j >= canvas.cols || y_j >= canvas.rows || x_j <= 0 || y_j <= 0)
	//	{
	//		continue;
	//	}
	//	maxSoFar = max(maxSoFar, (double) cannyOutput.at<uchar>(Point(x_j, y_j)));
	//}
	//return maxSoFar;
}

double avePfield(int i)
{
	assert(i >= 0 && i < mouseContour.size());
	int numRows = NEIGHBOURHOOD_SIZE * 2 + 1;
	int numNeighbours = pow(numRows, 2);
	double sumSoFar = 0;
	int countedSoFar = 0;

	for (int j = 0; j < numNeighbours; j++)
	{
		int x_j = (j % numRows) - 1 + mouseContour.at(i).x;
		int y_j = (j / numRows) - 1 + mouseContour.at(i).y;
		if (x_j >= canvas.cols || y_j >= canvas.rows || x_j <= 0 || y_j <= 0)
		{
			continue;
		}
		sumSoFar = sumSoFar + (double)cannyOutput.at<uchar>(Point(x_j, y_j));
		countedSoFar++;
	}
	return sumSoFar/countedSoFar;
}

double aveDistBtwnPts()
{
	int numVertices = mouseContour.size();
	if (numVertices < 4)
		return -1;
	double sumDist = hypot(mouseContour.at(0).x - mouseContour.at(numVertices - 1).x,
		mouseContour.at(0).y - mouseContour.at(numVertices - 1).y);
	for (int j = 1; j < numVertices; j++)
	{
		sumDist += hypot(mouseContour.at(j).x - mouseContour.at(j - 1).x,
			mouseContour.at(j).y - mouseContour.at(j - 1).y);
	}
	return sumDist / numVertices;
}

double maxMovingDist(int i, double aveDist)
{
	assert(i >= 0 && i < mouseContour.size());
	int numRows = NEIGHBOURHOOD_SIZE * 2 + 1;
	int numNeighbours = pow(numRows, 2);
	double maxSoFar;
	
	if (i > 0)
	{
		maxSoFar = abs(aveDist - pow(hypot(mouseContour.at(i).x - mouseContour.at(i - 1).x, 
										   mouseContour.at(i).y - mouseContour.at(i - 1).y), 2));
	}
	else
	{
		maxSoFar = abs(aveDist - pow(hypot(mouseContour.at(i).x - mouseContour.at(mouseContour.size() - 1).x, 
										   mouseContour.at(i).y - mouseContour.at(mouseContour.size() - 1).y), 2));
	}

	for (int j = 0; j < numNeighbours; j++)
	{
		int x_j = (j % numRows) - 1 + mouseContour.at(i).x;
		int y_j = (j / numRows) - 1 + mouseContour.at(i).y;
		if (x_j >= canvas.cols || y_j >= canvas.rows || x_j <= 0 || y_j <= 0)
		{
			continue;
		}
		if (i > 0)
		{
			maxSoFar = max(maxSoFar, pow(abs(aveDist - hypot( x_j - mouseContour.at(i - 1).x,
															  y_j - mouseContour.at(i - 1).y)), 2));
		}
		else
		{
			maxSoFar = max(maxSoFar, pow(abs(aveDist - hypot( x_j - mouseContour.at(mouseContour.size() - 1).x,
															  y_j - mouseContour.at(mouseContour.size() - 1).y)), 2));
		}
		
	}
	return maxSoFar;
}

double maxCurvature(int i)
{
	assert(i >= 0 && i < mouseContour.size());
	int numRows = NEIGHBOURHOOD_SIZE * 2 + 1;
	int numNeighbours = pow(numRows, 2);
	int size = mouseContour.size();
	double maxSoFar;
	if (i > 0)
	{
		maxSoFar = hypot(mouseContour.at(i - 1).x - (2 * mouseContour.at(i).x) + mouseContour.at((i + 1) % mouseContour.size()).x,
						 mouseContour.at(i - 1).y - (2 * mouseContour.at(i).y) + mouseContour.at((i + 1) % mouseContour.size()).y);
	}
	else
	{
		maxSoFar = hypot(mouseContour.at(size - 1).x - (2 * mouseContour.at(i).x) + mouseContour.at(i + 1).x,
						 mouseContour.at(size - 1).y - (2 * mouseContour.at(i).y) + mouseContour.at(i + 1).y);
	}

	for (int j = 0; j < numNeighbours; j++)
	{
		int x_j = (j % numRows) - 1 + mouseContour.at(i).x;
		int y_j = (j / numRows) - 1 + mouseContour.at(i).y;
		if (x_j >= canvas.cols || y_j >= canvas.rows || x_j <= 0 || y_j <= 0)
		{
			continue;
		}
		if (i > 0)
		{
			maxSoFar = max(maxSoFar, hypot(mouseContour.at(i - 1).x - (2 * mouseContour.at(i).x) + mouseContour.at((i + 1) % mouseContour.size()).x,
										   mouseContour.at(i - 1).y - (2 * mouseContour.at(i).y) + mouseContour.at((i + 1) % mouseContour.size()).y));
		}
		else if (i == 0)
		{
			maxSoFar = max(maxSoFar, hypot(mouseContour.at(size - 1).x - (2 * mouseContour.at(i).x) + mouseContour.at(i + 1).x,
										   mouseContour.at(size - 1).y - (2 * mouseContour.at(i).y) + mouseContour.at(i + 1).y));
		}

	}
	return maxSoFar;
}


void avePointsThreshCallback(int, void*)
{
	AVE_DIST_CHANGE_THRESHOLD = distPercentage / (double)10000;
}

void cannyThreshCallback(int, void*)
{
	Canny(canvasGrey, cannyUnblurred, CANNY_THRESHOLD, CANNY_THRESHOLD * 2, 3);
	blur(cannyUnblurred, cannyOutput, Size(AMOUNT_OF_BLUR, AMOUNT_OF_BLUR));
	imshow("CANNY", cannyOutput);
}

void blurCallback(int, void*)
{
	if (AMOUNT_OF_BLUR <= 0)
	{
		AMOUNT_OF_BLUR = 1;
	}
		
	blur(cannyUnblurred, cannyOutput, Size(AMOUNT_OF_BLUR, AMOUNT_OF_BLUR));
	imshow("CANNY", cannyOutput);
}
