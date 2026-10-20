#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

using namespace std;
using namespace cv;

const int TAM_BLUR = 65;

int main()
{
    vector<int> imagens = {60, 82, 114, 150, 205};

    for (int qtd : imagens)
    {
        string nome_arquivo = to_string(qtd) + ".bmp";

        Mat img = imread(nome_arquivo, IMREAD_GRAYSCALE);

        if (img.empty())
        {
            cout << "Erro ao carregar " << nome_arquivo << endl;
            continue;
        }

        img.convertTo(img, CV_32F, 1.0 / 255.0);

        Mat blur_img;
        blur_img = Mat::zeros(img.size(), img.type());
        blur(img, blur_img, Size(TAM_BLUR, TAM_BLUR));

        Mat dif = img - blur_img;
        normalize(dif, dif, 0, 255, NORM_MINMAX, CV_8U);

        Mat buffer;
        threshold(dif, buffer, 0, 255, THRESH_BINARY | THRESH_OTSU);

        Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
        Mat binary;
        morphologyEx(buffer, binary, MORPH_OPEN, kernel);
        imshow("Binaria - " + to_string(qtd), binary);

        Mat dist;
        distanceTransform(binary, dist, DIST_L2, 5);

        Mat dist_visualizacao;
        normalize(dist, dist_visualizacao, 0, 255, NORM_MINMAX, CV_8U);
        imshow("Distancia do centro - " + to_string(qtd), dist_visualizacao);

        kernel = getStructuringElement(MORPH_CROSS, Size(3, 3));
        Mat dilatada;
        dilate(dist, dilatada, kernel);

        Mat resultado;
        resultado = dilatada - dist - dist;
        imshow("resultado - " + to_string(qtd), resultado);

        waitKey(0);
        destroyAllWindows();
    }

    return 0;
}
