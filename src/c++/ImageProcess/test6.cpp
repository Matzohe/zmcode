#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>

using namespace std;

// LZW Encoding Function
vector<int> lzwEncode(const vector<vector<int> >& src, int M, int N) {
    // Flatten the 2D image to a 1D vector (row-major order)
    vector<int> data;

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            data.push_back(src[i][j]);
        }
    }

    // Initialize the dictionary with single-character sequences
    unordered_map<string, int> dictionary;
    int dictSize = 256; // Since gray levels are in range [0,255]
    for (int i = 0; i < 256; ++i) {
        dictionary[string(1, i)] = i;
    }

    string currentStr;
    vector<int> encodedData;

    for (int pixel : data) {
        char pixelChar = static_cast<char>(pixel);
        string newStr = currentStr + pixelChar;

        if (dictionary.count(newStr)) {
            currentStr = newStr;
        } else {
            encodedData.push_back(dictionary[currentStr]);
            dictionary[newStr] = dictSize++;
            currentStr = string(1, pixelChar);
        }
    }

    // Output the last code
    if (!currentStr.empty()) {
        encodedData.push_back(dictionary[currentStr]);
    }

    return encodedData;
}

int main() {
    // Input dimensions
    int M, N;
    cin >> M;
    cin >> N;

    // Input the gray-scale image
    vector<vector<int> > src(M, vector<int>(N));
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            cin >> src[i][j];
        }
    }

    // Perform LZW encoding
    vector<int> encodedData = lzwEncode(src, M, N);
    // Output the encoded data
    for (size_t i = 0; i < encodedData.size(); ++i) {
        cout << encodedData[i];
        if (i < encodedData.size() - 1) {
            cout << " ";
        }
    }
    cout << endl;

    return 0;
}
