#include "utils/RollingHash.h"
#include "utils/HuffmanTree.h"
#include <fstream>
#include <string>
#include <sstream>
#include <iostream>
#include <chrono>

int main(){

    std::ifstream file("/Users/a1/PythonProgram/zmcode/testDataset/caffe_ilsvrc12.zip");
    // 设置文件读写位置
    // 分多段进行压缩和解压缩
    file.seekg(0, std::ios::end);
    std::streampos size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::streampos position = 0;
    std::vector<char> buffer(size);
    file.seekg(position, std::ios::beg);
    file.read(buffer.data(), size);
    std::string input_str(buffer.begin(), buffer.end());

    std::cout << input_str.size() << std::endl;
    auto start = std::chrono::steady_clock::now();

    RollingHash rollingHash(3);
    rollingHash.RollingHashProcess(input_str);

    auto end = std::chrono::steady_clock::now();
    std::cout << "compression Time cost = " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "[ms]" << std::endl;

    start = std::chrono::steady_clock::now();

    std::string filePath = "example.txt";
    std::ifstream another_file(filePath);
    std::stringstream buffer_save;
    buffer_save << another_file.rdbuf();
    another_file.close();
    std::string output_str = buffer_save.str();
    std::cout << output_str.size() << std::endl;
    std::string decoder_2 = rollingHash.DecodeLZ77WithInfo(output_str);
    std::ofstream decoded_file("OneMarkdown.zip", std::ios::app);
    decoded_file << decoder_2;
    decoded_file.close();

    end = std::chrono::steady_clock::now();
    std::cout << "decompression Time cost = " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "[ms]" << std::endl;
    return 0;
}