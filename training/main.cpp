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

vector<cv::String> readFilesInFolder(string path){
  vector<cv::String> files;
  cv::utils::fs::glob(path, "*.jpg" , files, true, true);
  return files;
}

int main( int argc, char* argv[] ){

    Ptr<LBPHFaceRecognizer> fr = LBPHFaceRecognizer::create();
    
    string path = "../image_catching/";
    vector<string> files = readFilesInFolder(path);
    vector<string> labels; 
    map<string, int> dic = {};
    vector<Mat> train_images;
    vector<int> train_indexes;
    
    int currentId = 0;
    for (auto file : files) {

      string parent = cv::utils::fs::getParent(file);
      string label = parent.substr(parent.find_last_of("/") + 1, parent.length());
      /// cout << file << " " << parent << "\n " ;
      cout << label << " " << dic[label] <<"\n " ;
      
      
      if(dic[label] == 0){
        currentId++;
        dic[label] = currentId;
      }

      Mat img = imread(file, IMREAD_GRAYSCALE);

      int id_ = dic[label];
      train_images.push_back(img);
      train_indexes.push_back(id_);
    
    }

    ofstream file;
    file.open ("../data/struct.txt");
 
    for (auto & pair : dic){
      file << pair.first << " " << pair.second << '\n';
    }

    file.close();
    cout << endl;

    cout << "start training \n";
    fr -> train(train_images, train_indexes);
    fr -> save("../data/uniquer.xml");
    cout << "end training \n";
    return 0;
}