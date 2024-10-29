#include <iostream>
#include <windows.h>
#include <string>
#include <queue>
#include <ctime>
#include <cstdlib>
#include <thread>

#define BUFFER_SIZE 6
#define MAX_STRING_LENGTH 10

using namespace std;

HANDLE mutex;  // 互斥量
HANDLE empty;  // 信号量，表示空缓冲区
HANDLE full;   // 信号量，表示满缓冲区

queue<string> buffer;  // 缓冲池

void producer(int id) {
    for (int i = 0; i < 12; i++) {
        Sleep(rand() % 1000); // 随机等待时间
        string data = "Data_" + to_string(id) + "_" + to_string(i);

        WaitForSingleObject(empty, INFINITE);  // 等待空缓冲区
        WaitForSingleObject(mutex, INFINITE);  // 进入临界区

        buffer.push(data);
        cout << "[Producer " << id << "] Added: " << data << " at " << time(0) << ", Buffer size: " << buffer.size() << endl;

        ReleaseMutex(mutex);          // 离开临界区
        ReleaseSemaphore(full, 1, 0); // 增加满缓冲区信号量
    }
}

void consumer(int id) {
    for (int i = 0; i < 8; i++) {
        Sleep(rand() % 1000); // 随机等待时间

        WaitForSingleObject(full, INFINITE);   // 等待满缓冲区
        WaitForSingleObject(mutex, INFINITE);  // 进入临界区

        string data = buffer.front();
        buffer.pop();
        cout << "[Consumer " << id << "] Removed: " << data << " at " << time(0) << ", Buffer size: " << buffer.size() << endl;

        ReleaseMutex(mutex);           // 离开临界区
        ReleaseSemaphore(empty, 1, 0); // 增加空缓冲区信号量
    }
}

int main() {
    srand((unsigned)time(NULL)); // 初始化随机种子

    // 初始化信号量和互斥量
    mutex = CreateMutex(NULL, FALSE, NULL);
    empty = CreateSemaphore(NULL, BUFFER_SIZE, BUFFER_SIZE, NULL);
    full = CreateSemaphore(NULL, 0, BUFFER_SIZE, NULL);

    // 创建生产者线程
    thread producer1(producer, 1);
    thread producer2(producer, 2);

    // 创建消费者线程
    thread consumer1(consumer, 1);
    thread consumer2(consumer, 2);
    thread consumer3(consumer, 3);

    // 等待所有线程完成
    producer1.join();
    producer2.join();
    consumer1.join();
    consumer2.join();
    consumer3.join();

    // 释放资源
    CloseHandle(mutex);
    CloseHandle(empty);
    CloseHandle(full);

    return 0;
}
