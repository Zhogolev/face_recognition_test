#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/face.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <sys/time.h>
#include <ctime>
#include <opencv2/core/utils/filesystem.hpp>
    
using namespace std;
using namespace cv;
using namespace face;

// Function for Face Detection
void detectAndSave( Mat& img, CascadeClassifier& cascade, double scale, char* user);

  int detected = 0;
int main( int argc, char* argv[] )
{   
    

    if(argc == 1){
        cout << "Name of user should be declared";
        return -1;
    }
    char* user = argv[1];

    if(!cv::utils::fs::exists("./faces/" + string(user))){
        cv::utils::fs::createDirectory("./faces/" + string(user));
    }
    
    // VideoCapture class for playing video for which faces to be detected
    VideoCapture capture; 
    Mat smallImg, image;
   
    // PreDefined trained XML classifiers with facial features
    CascadeClassifier cascade; 
    double scale=1;
    cascade.load( "../data/haarcascades_cuda/haarcascade_frontalface_alt2.xml" ) ; 
  
    // Start Video..1) 0 for WebCam 2) "Path to Video" for a Local Video
    capture.open(0); 
    if( capture.isOpened() )
    {
        // Capture smallImgs from video and detect faces
        cout << "Face Detection Started...." << endl;
        while(1 && detected < 100)
        {
            capture >> smallImg;
            if( smallImg.empty() )
                break;
            Mat smallImg1 = smallImg.clone();
            
            detectAndSave( smallImg1, cascade, scale , user); 
            
            
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
uint64_t timeSinceEpochMillisec() {
  using namespace std::chrono;
  return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

uint64_t last_detected = timeSinceEpochMillisec();

void detectAndSave( Mat& img, CascadeClassifier& cascade,
                    double scale, char* user)
{
    vector<Rect>  faces;
    Mat gray;
  
    cvtColor( img, gray, COLOR_BGR2GRAY ); // Convert to Gray Scale
    double fx = 1 / scale;
  
    cascade.detectMultiScale( gray, faces, 1.1, 
                            2, 0|CASCADE_SCALE_IMAGE, Size(96, 96) );

    int size_faces = faces.size();
    for (int i = 0; i < size_faces; ++i){
        uint64_t now_time = timeSinceEpochMillisec();
        cout<< "now_time" << now_time  << '\n';
        cout<< "last_time" << last_detected  << '\n';
        if(now_time < last_detected + 300){
            continue;
        }
        last_detected = now_time;
        Rect current = faces[i];

        int midX = (2 * current.x + current.width) / 2;
        int midY = (2 * current.y + current.height) / 2;

        int sourceWidth = img.cols;
        int sourceHeight = img.rows;

        int width = current.width;
        int height = current.height;
        int size = max(width, height);
        size = size * 1.5;
        int halfSize = size / 2;
        
        if(midX + halfSize < sourceWidth && midX - halfSize > 0 && midY - halfSize> 0 && midY + halfSize < sourceWidth){
            Rect rect = Rect(Point(midX - halfSize, midY - halfSize), Size(size, size));
            Mat cropped = img(current);

            cout << img.cols << " " << img.rows << "\n";
            bool contain = rect.contains(Point(current.x, current.y)) && 
            rect.contains(Point(current.x + current.width, current.y + current.height));
            detected ++;
            string imgPath = "./faces/" + string(user) + "/" + to_string(detected) + ".jpg";
           
        
            Mat resized;
            resize(img(rect), resized, Size(96,96),INTER_LINEAR);
            imwrite(imgPath, resized);
        }

        

     }    
    
    imshow( "Face Detection", img ); 

}