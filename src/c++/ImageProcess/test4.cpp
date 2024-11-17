#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>

using namespace std;

// 定义方向数组用于遍历4邻域和8邻域
int dx4[4] = {-1, 1, 0, 0};
int dy4[4] = {0, 0, -1, 1};

int dx8[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
int dy8[8] = {-1, 0, 1, -1, 1, -1, 0, 1};

// 坐标点
struct Point {
    int x, y;
    Point(int x, int y) : x(x), y(y) {}
};

// 比较函数，用于按照y优先、x次优排序
bool comparePoints(const Point &a, const Point &b) {
    if (a.y == b.y) return a.x < b.x;
    return a.y < b.y;
}

void detectConnectedComponents(vector<vector<int>> &image, int w, int h, int k) {
    vector<vector<bool>> visited(h, vector<bool>(w, false));
    vector<vector<Point>> components;
    int directions = (k == 4) ? 4 : 8;
    int *dx = (k == 4) ? dx4 : dx8;
    int *dy = (k == 4) ? dy4 : dy8;

    // 遍历图像
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            if (image[i][j] == 1 && !visited[i][j]) {
                // 开始一个新的连通域
                vector<Point> component;
                queue<Point> q;
                q.push(Point(j, i)); // 注意顺序：x为列，y为行
                visited[i][j] = true;

                // 广度优先搜索
                while (!q.empty()) {
                    Point p = q.front();
                    q.pop();
                    component.push_back(p);

                    // 遍历邻域
                    for (int d = 0; d < directions; ++d) {
                        int nx = p.x + dx[d];
                        int ny = p.y + dy[d];
                        if (nx >= 0 && nx < w && ny >= 0 && ny < h && !visited[ny][nx] && image[ny][nx] == 1) {
                            visited[ny][nx] = true;
                            q.push(Point(nx, ny));
                        }
                    }
                }

                // 对连通域内像素按照y优先、x次优排序
                sort(component.begin(), component.end(), comparePoints);
                components.push_back(component);
            }
        }
    }

    // 输出结果
    for (size_t i = 0; i < components.size(); ++i) {
        cout << i + 1 << "-th component:" << endl;
        for (const auto &p : components[i]) {
            cout << p.x << " " << p.y << endl;
        }
    }
}

int main() {
    int w, h, k;
    cin >> w >> h >> k;

    vector<vector<int>> image(h, vector<int>(w));
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            cin >> image[i][j];
        }
    }

    detectConnectedComponents(image, w, h, k);

    return 0;
}
