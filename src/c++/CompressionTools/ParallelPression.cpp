#include "utils/RollingHash.h"
#include "utils/HuffmanTree.h"
#include <fstream>
#include <string>
#include <sstream>
#include <iostream>
#include <chrono>
#include <numeric>
#include <omp.h>


int main(){

    std::ifstream file("/Users/a1/PythonProgram/zmcode/testDataset/caffe_ilsvrc12.zip");
    // 设置文件读写位置
    // 分多段进行压缩和解压缩

    int max_size_ = 512 * 1024;
    

    file.seekg(0, std::ios::end);
    std::streampos size = file.tellg();
    file.close();
    int file_length = static_cast<int>(size);
    int part = file_length / max_size_ + 1;

    std::vector<std::string> output_strs(part, "");
    std::vector<std::string> decoding_info(part, "");

    auto start = std::chrono::steady_clock::now();
    #pragma omp parallel shared(output_strs, decoding_info)
    {
        #pragma omp parallel for
        for(int i = 0; i < part; i++){
            // 每个线程都有自己的文件流
            std::ifstream file("/path/to/your/file.zip", std::ios::binary);
            std::string data_input;
            if (i == part - 1){
                file.seekg(i * max_size_, std::ios::beg);
                int length = file_length - i * max_size_;
                std::vector<char> buffer(length);
                file.read(buffer.data(), length);
                data_input = std::string(buffer.begin(), buffer.end());
            }
            else{
                file.seekg(i * max_size_, std::ios::beg);
                std::vector<char> buffer(max_size_);
                file.read(buffer.data(), max_size_);
                data_input = std::string(buffer.begin(), buffer.end());
            }
            file.close();
            RollingHash rollingHash(3);
            output_strs[i] = rollingHash.RollingHashProcess(data_input);
            decoding_info[i] = rollingHash.getDecodeInfo();
        }
    }
    auto end = std::chrono::steady_clock::now();
    std::cout << "compression Time cost = " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "[ms]" << std::endl;

    std::string encoded_string = std::accumulate(output_strs.begin(), output_strs.end(), std::string());
    std::string decode_info = std::accumulate(decoding_info.begin(), decoding_info.end(), std::string());

    std::ofstream current_file("output.txt", std::ios::app);
    current_file << encoded_string;
    current_file.close();
    return 0;
}