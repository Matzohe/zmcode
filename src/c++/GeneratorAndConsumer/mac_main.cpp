#include <iostream>
#include <pthread.h>
#include <unistd.h>
#include <string>
#include <queue>
#include <cstdlib>
#include <ctime>

#define BUFFER_SIZE 6
#define MAX_STRING_LENGTH 10

using namespace std;

pthread_mutex_t _mutex;           // 互斥锁
pthread_cond_t buffer_not_full;  // 条件变量，表示缓冲区不满
pthread_cond_t buffer_not_empty; // 条件变量，表示缓冲区不空

queue<string> buffer;  // 缓冲池

void* producer(void* arg) {
    int id = *(int*)arg;
    for (int i = 0; i < 12; i++) {
        sleep(rand() % 10); // 随机等待时间
        string data = "Data_" + to_string(id) + "_" + to_string(i);

        pthread_mutex_lock(&_mutex); // 加锁

        while (buffer.size() == BUFFER_SIZE) {
            pthread_cond_wait(&buffer_not_full, &_mutex); // 等待缓冲区不满
        }

        buffer.push(data);
        cout << "[Producer " << id << "] Added: " << data << " at " << time(0) << ", Buffer size: " << buffer.size() << endl;

        pthread_cond_signal(&buffer_not_empty); // 通知消费者缓冲区不空
        pthread_mutex_unlock(&_mutex); // 解锁
    }
    return NULL;
}

void* consumer(void* arg) {
    int id = *(int*)arg;
    for (int i = 0; i < 8; i++) {
        sleep(rand() % 10); // 随机等待时间

        pthread_mutex_lock(&_mutex); // 加锁

        while (buffer.empty()) {
            pthread_cond_wait(&buffer_not_empty, &_mutex); // 等待缓冲区不空
        }

        string data = buffer.front();
        buffer.pop();
        cout << "[Consumer " << id << "] Removed: " << data << " at " << time(0) << ", Buffer size: " << buffer.size() << endl;

        pthread_cond_signal(&buffer_not_full); // 通知生产者缓冲区不满
        pthread_mutex_unlock(&_mutex); // 解锁
    }
    return NULL;
}

int main() {
    
    srand((unsigned)time(NULL)); // 初始化随机种子

    // 初始化互斥锁和条件变量
    pthread_mutex_init(&_mutex, NULL);
    pthread_cond_init(&buffer_not_full, NULL);
    pthread_cond_init(&buffer_not_empty, NULL);

    // 创建生产者线程
    pthread_t producers[2];
    int producer_ids[2] = {1, 2};
    for (int i = 0; i < 2; i++) {
        pthread_create(&producers[i], NULL, producer, &producer_ids[i]);
    }

    // 创建消费者线程
    pthread_t consumers[3];
    int consumer_ids[3] = {1, 2, 3};
    for (int i = 0; i < 3; i++) {
        pthread_create(&consumers[i], NULL, consumer, &consumer_ids[i]);
    }

    // 等待所有线程完成
    for (int i = 0; i < 2; i++) {
        pthread_join(producers[i], NULL);
    }
    for (int i = 0; i < 3; i++) {
        pthread_join(consumers[i], NULL);
    }

    // 销毁互斥锁和条件变量
    pthread_mutex_destroy(&_mutex);
    pthread_cond_destroy(&buffer_not_full);
    pthread_cond_destroy(&buffer_not_empty);

    return 0;
}
