#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define INPUT_IMAGE "60.bmp"

int main()
{
    Mat img = imread(INPUT_IMAGE, IMREAD_GRAYSCALE);
    if (!img.data)
    {
        cerr << "Erro abrindo a imagem.\n";
        exit(1);
    }

    img.convertTo(img, CV_32F, 1.0 / 255.0);

    imshow("01 - binarizada", img);
    imwrite("01 - binarizada.png", img);
    waitKey();

    return (0);
}