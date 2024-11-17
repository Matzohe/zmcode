#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

// 双线性插值函数
int bilinearInterpolate(const vector<vector<int> >& src, double x, double y) {
    int h = src.size();
    int w = src[0].size();

    // 四个邻近像素点的坐标
    int x1 = floor(x);
    int x2 = x1 + 1;
    int y1 = floor(y);
    int y2 = y1 + 1;

    if (x1 == x && y1 == y) {
        x1 = max(0, min(x1, w - 1));
        y1 = max(0, min(y1, h - 1));
        return round(src[y1][x1]);
    }


    // 边界检查
    x1 = max(0, min(x1, w - 1));
    x2 = max(0, min(x2, w - 1));
    y1 = max(0, min(y1, h - 1));
    y2 = max(0, min(y2, h - 1));

    // 四个点的像素值
    double Q11 = src[y1][x1];
    double Q21 = src[y1][x2];
    double Q12 = src[y2][x1];
    double Q22 = src[y2][x2];

    // 插值公式
    double R1 = ((x2 - x) * Q11 + (x - x1) * Q21);
    double R2 = ((x2 - x) * Q12 + (x - x1) * Q22);
    double P = ((y2 - y) * R1 + (y - y1) * R2);

    return round(P);
}

// 图像缩放函数
vector<vector<int> > resizeImage(const vector<vector<int> >& src, int newHeight, int newWidth) {
    int oldHeight = src.size();
    int oldWidth = src[0].size();

    vector<vector<int> > dst(newHeight, vector<int>(newWidth, 0));

    // 计算缩放比例
    double scaleY = static_cast<double>(oldHeight) / newHeight;
    double scaleX = static_cast<double>(oldWidth) / newWidth;

    // 中心化
    double srcY_ = (oldHeight - 1) / 2.0;
    double srcX_ = (oldWidth - 1) / 2.0;
    double new_srcY = (newHeight - 1) / 2.0;
    double new_srcX = (newWidth - 1) / 2.0;


    // 遍历新图像中的每个像素
    for (int i = 0; i < newHeight; ++i) {
        for (int j = 0; j < newWidth; ++j) {
            // 找到对应的原图坐标
            double srcY = (i - new_srcY) * scaleY + srcY_;
            double srcX = (j - new_srcX) * scaleX + srcX_;

            // 插值计算
            dst[i][j] = bilinearInterpolate(src, srcX, srcY);
        }
    }

    return dst;
}

// 打印图像
void printImage(const vector<vector<int> >& image) {
    for (const auto& row : image) {
        for (int j = 0; j < row.size(); ++j) {
            if (j < row.size() - 1) cout << row[j] << " ";
            else cout << row[j];
        }
        cout << endl;
    }
}

int main() {
    int oldHeight, oldWidth;
    cin >> oldHeight >> oldWidth;

    int newHeight, newWidth;
    cin >> newHeight >> newWidth;

    // 读取图像
    vector<vector<int> > src(oldHeight, vector<int>(oldWidth));

    for (int i = 0; i < oldHeight; ++i) {
        for (int j = 0; j < oldWidth; ++j) {
            cin >> src[i][j];
            if (src[i][j] > 255){
                src[i][j] = 255;
            }
        }
    }

    // 缩放图像
    vector<vector<int> > dst = resizeImage(src, newHeight, newWidth);

    // 打印结果
    printImage(dst);

    return 0;
}
