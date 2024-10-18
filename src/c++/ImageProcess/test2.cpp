#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

vector<int> histogram_equalization(vector<vector<int>> image) {
    int height = 8;
    int width = 8;
    int gray_levels = 8;
    vector<int> histogram(gray_levels, 0);
    vector<int> cdf(gray_levels, 0);
    vector<int> mapping(gray_levels, 0);

    // Calculate the histogram
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            histogram[image[i][j]]++;
        }
    }

    // Calculate the cumulative distribution function (CDF)
    cdf[0] = histogram[0];
    for (int i = 1; i < gray_levels; ++i) {
        cdf[i] = cdf[i - 1] + histogram[i];
    }

    // Calculate the mapping using CDF
    for (int i = 0; i < gray_levels; ++i) {
        float output = float(cdf[i] * (gray_levels - 1)) / float(height * width);
        mapping[i] = int(output + 0.5);
    }

    return mapping;
}

int main() {
    vector<vector<int>> image(8, vector<int>(8));

    // Input 8x8 grayscale image
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            cin >> image[i][j];
        }
    }

    // Perform histogram equalization
    vector<int> mapping = histogram_equalization(image);

    // Output the equalized gray level mapping
    for (int i = 0; i < mapping.size(); ++i) {
        cout << mapping[i];
        if (i < mapping.size() - 1) {
            cout << " ";
        }
    }
    cout << endl;

    return 0;
}
