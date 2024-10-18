#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

vector<vector<int>> convolve(const vector<vector<int>>& src, const vector<vector<double>>& kernel) {
    int m = src.size();
    int n = src[0].size();
    int u = kernel.size();
    int v = kernel[0].size();
    
    int pad_h = u / 2;
    int pad_w = v / 2;
    vector<vector<int>> result(m, vector<int>(n, 0));

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int ki = 0; ki < u; ++ki) {
                for (int kj = 0; kj < v; ++kj) {
                    int src_i = i + ki - pad_h;
                    int src_j = j + kj - pad_w;
                    if (src_i >= 0 && src_i < m && src_j >= 0 && src_j < n) {
                        sum += src[src_i][src_j] * kernel[ki][kj];
                    }
                }
            }
            result[i][j] = static_cast<int>(round(sum));
        }
    }
    return result;
}

int main() {
    int m, n, u, v;
    cin >> m >> n;
    cin >> u >> v;

    vector<vector<int>> src(m, vector<int>(n));
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            cin >> src[i][j];
        }
    }

    vector<vector<double>> kernel(u, vector<double>(v));
    for (int i = 0; i < u; ++i) {
        for (int j = 0; j < v; ++j) {
            cin >> kernel[i][j];
        }
    }

    vector<vector<int>> result = convolve(src, kernel);
    
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            cout << result[i][j];
            if (j < n) {
                cout << " ";
            }
        }
        cout << "\n";
    }

    return 0;
}
