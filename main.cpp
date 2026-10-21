#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <random>

using namespace std;
using namespace cv;

const int TAM_BLUR = 65;
const double LIMIAR_DIST = 0.42;

int main()
{
    vector<int> imagens = {60, 82, 114, 150, 205};
    RNG rng(67);

    for (int qtd : imagens)
    {
        string nome_arquivo = to_string(qtd) + ".bmp";
        Mat img = imread(nome_arquivo, IMREAD_GRAYSCALE);
        if (img.empty())
        {
            cout << "Erro ao carregar " << nome_arquivo << endl;
            continue;
        }

        Mat original = img.clone();
        img.convertTo(img, CV_32F, 1.0 / 255.0);

        Mat blur_img;
        blur(img, blur_img, Size(TAM_BLUR, TAM_BLUR));

        Mat dif = img - blur_img;
        normalize(dif, dif, 0, 255, NORM_MINMAX, CV_8U);

        Mat binary;
        threshold(dif, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);

        Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
        morphologyEx(binary, binary, MORPH_OPEN, kernel);
        morphologyEx(binary, binary, MORPH_CLOSE, kernel);

        Mat dist;
        distanceTransform(binary, dist, DIST_L2, 5);

        double max_dist;
        minMaxLoc(dist, nullptr, &max_dist);

        Mat centros;
        threshold(dist, centros, LIMIAR_DIST * max_dist, 255, THRESH_BINARY);
        centros.convertTo(centros, CV_8U);

        Mat fundos;
        dilate(binary, fundos, kernel, Point(-1, -1), 3);

        Mat meio_termo;
        subtract(fundos, centros, meio_termo);

        Mat labels;
        connectedComponents(centros, labels);
        labels += 1;
        labels.setTo(0, meio_termo == 255);

        Mat img_watershed;
        cvtColor(original, img_watershed, COLOR_GRAY2BGR);

        watershed(img_watershed, labels);

        Mat colorida(original.size(), CV_8UC3, Scalar(0, 0, 0));

        double min_label, max_label;
        minMaxLoc(labels, &min_label, &max_label);

        int total_graos = 0;
        for (int label = 2; label <= (int)max_label; label++)
        {
            Mat regiao = (labels == label);

            total_graos++;

            Scalar cor(
                rng.uniform(67, 256),
                rng.uniform(67, 256),
                rng.uniform(67, 256));

            colorida.setTo(cor, regiao);
        }

        cout << "Imagem: " << nome_arquivo << " | Graos detectados: " << total_graos << endl;
        imshow("Graos coloridos - " + to_string(qtd), colorida);

        waitKey(0);
        destroyAllWindows();
    }

    return 0;
}