Object Tracking with the Attractable Greedy Snake
=================================================
This project implements the Attractable Greedy Snake according to the following papers:

Williams, Donna J., and Mubarak Shah. "A Fast Algorithm for Active Contours and Curvature Estimation." CVGIP: Image Understanding (1992): 14-26. ACM Digital Library. ACM. Web. 6 Dec. 2014. <http://dl.acm.org/citation.cfm?id=134401>.

Ji, Lilian, and Hong Yan. "Attractable Snakes Based on the Greedy Algorithm for Contour Extraction." Pattern Recognition (2002): 791-806. Science Direct. Elsevier. Web. 6 Dec. 2014. <http://www.sciencedirect.com/science/article/pii/S0031320301000851>.

How to Use
==========
To switch between static image and video modes, comment or uncomment the line:
	#define STATIC_IMAGE_MODE 
	
To change the input video, change the path in this line:
	VideoCapture cap(<PATH TO VIDEO>);

To change the input image, change the path in this line:
	canvas = imread("<PATH TO IMAGE>");

The name of the output video (used in the demo) is:
	OutputVideo.avi
	(Unfortunately, the video quality seems to be kind of terrible)

