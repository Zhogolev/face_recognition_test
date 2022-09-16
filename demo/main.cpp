#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/face.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/face/facerec.hpp>
#include <iostream>
#include <sys/time.h>
#include <ctime>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/face.hpp>
#include <map>
#include <fstream>

using namespace std;
using namespace cv;
using namespace face;

vector<string> readNames(){
  vector<string> data;
  string line;
  ifstream file ("../data/struct.txt");
  if (file.is_open())
  {
    while ( getline (file,line) )
    {
      data.push_back(line);
    }
    file.close();
  }
  return data;
}


int main( int argc, char* argv[] ){
    VideoCapture capture;
    Mat smallImg, image;
    vector<string> names = readNames();
    CascadeClassifier cascade; 
    double scale=1;
    cascade.load( "../data/haarcascades_cuda/haarcascade_frontalface_alt2.xml" ) ; 

    Ptr<LBPHFaceRecognizer> fr = LBPHFaceRecognizer::create();
    fr -> read("../data/uniquer.xml");

    capture.open(0); 
    if( capture.isOpened() )
    {
        // Capture smallImgs from video and detect faces
        cout << "Face Detection Started...." << endl;
        while(1)
        {
            capture >> smallImg;
            if( smallImg.empty() )
                break;
            Mat smallImg1 = smallImg.clone();
            
            vector<Rect> faces;
            Mat gray;
            
            cvtColor( smallImg1, gray, COLOR_BGR2GRAY ); // Convert to Gray Scale
            double fx = 1 / scale;
            
              cascade.detectMultiScale( gray, faces, 1.1, 
                                      2, 0|CASCADE_SCALE_IMAGE, Size(96, 96) );

              for (Rect face : faces){
                  rectangle(smallImg, face,  Scalar(255, 255, 0), 1, 8, 0);
                  int label;
                  double confiedence; 
                  fr -> predict(gray(face), label, confiedence);
                  if(label != -1 && label <= names.size()){
                    putText(smallImg, names[label -1], face.tl(), cv::FONT_HERSHEY_PLAIN, 1, Scalar(255, 255, 0));
                  }
                  cout << "label: "<< label << " confidence: " << confiedence << "" "\n";
              }

            imshow( "Face Detection", smallImg ); 
            
            
            char c = (char)waitKey(10);
          
            // Press q to exit from window
            if( c == 27 || c == 'q' || c == 'Q' ) 
                break;
        }
    }
    else
        cout<<"Could not Open Camera";
    return 0;
}
