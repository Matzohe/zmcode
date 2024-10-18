#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

vector<vector<int>> blur(vector<vector<int>> src, int height_filter, int width_filter) {
    int height_src = src.size();
    int width_src = src[0].size();
    vector<vector<int>> dst(height_src, vector<int>(width_src, 0));

    int hf_h = height_filter / 2;
    int hf_w = width_filter / 2;

    for (int i = 0; i < height_src; ++i) {
        for (int j = 0; j < width_src; ++j) {
            long long sum = 0;
            int count = 0;
            for (int m = -hf_h; m <= hf_h; ++m) {
                for (int n = -hf_w; n <= hf_w; ++n) {
                    int y = i + m;
                    int x = j + n;
                    if (y >= 0 && y < height_src && x >= 0 && x < width_src) {
                        sum += src[y][x];
                        count++;
                    }
                }
            }
            float output = float(sum) / float(height_filter * width_filter);
            dst[i][j] = int(output + 0.5);
        }
    }

    return dst;
}

int main() {
    int height_src, width_src, height_filter, width_filter;
    cin >> height_src >> width_src >> height_filter >> width_filter;

    vector<vector<int>> src(height_src, vector<int>(width_src));
    for (int i = 0; i < height_src; ++i) {
        for (int j = 0; j < width_src; ++j) {
            cin >> src[i][j];
        }
    }

    vector<vector<int>> dst = blur(src, height_filter, width_filter);

    for (int i = 0; i < height_src; ++i) {
        for (int j = 0; j < width_src; ++j) {
            cout << dst[i][j] << " ";
        }
        cout << "\n";
    }

    return 0;
}
