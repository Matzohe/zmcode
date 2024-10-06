#include "utils/RollingHash.h"
#include "utils/HuffmanTree.h"
#include <fstream>
#include <string>
#include <sstream>
#include <iostream>

int main(){

    std::ifstream file("/Users/a1/C_Program/个人代码+毛子鸣.zip");
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


    RollingHash rollingHash(3);
    rollingHash.RollingHashProcess(input_str);

    std::string filePath = "example.txt";
    std::ifstream another_file(filePath);
    std::stringstream buffer_save;
    buffer_save << another_file.rdbuf();
    another_file.close();
    std::string output_str = buffer_save.str();
    std::cout << output_str.size() << std::endl;
    std::string decoder_2 = rollingHash.DecodeLZ77WithInfo(output_str);
    std::ofstream decoded_file("个人代码+毛子鸣.zip", std::ios::app);
    decoded_file << decoder_2 << std::endl;
    decoded_file.close();
    // std::unordered_map<unsigned int, std::string> huffmanCode;
    // HuffmanTree huffmanTree(huffmanCode, nullptr);
    // std::unordered_map<unsigned int, long long> charcount = rollingHash.getCharcount(output_str);

    // std::cout << charcount.size() << std::endl;

    // huffmanTree.initializeHuffmanTree(charcount);

    // huffmanTree.encode(input_str);
    
    // std::ifstream encoded_file("new_example.txt");
    // std::stringstream buffer_encode;
    // buffer_encode << encoded_file.rdbuf();
    // encoded_file.close();
    // std::string encoded_str = buffer_encode.str();
    // std::string decoder_1 = huffmanTree.decode(encoded_str);
    // std::ofstream decoded_file("output.txt", std::ios::app);
    // decoded_file << decoder_1 << std::endl;
    // decoded_file.close();


    // std::string decoder_2 = rollingHash.DecodeLZ77WithInfo(decoder_1);
    // std::ofstream decoded_file("output.txt", std::ios::app);
    // decoded_file << decoder_2 << std::endl;
    // decoded_file.close();
    return 0;
}