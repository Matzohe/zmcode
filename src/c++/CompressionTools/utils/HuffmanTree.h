#ifndef HUFFMANTREE_H
#define HUFFMANTREE_H

#include <iostream>
#include <unordered_map>
#include <queue>
#include <bitset>
#include <fstream>

struct HuffmanNode{
    unsigned int ch;
    int freq;
    HuffmanNode* left;
    HuffmanNode* right;
    HuffmanNode(unsigned int character, int frequency) : ch(character), freq(frequency), left(nullptr), right(nullptr) {}
};

class HuffmanTree{
private:
    std::unordered_map<unsigned int, std::string>& huffmanCode;
    HuffmanNode *MyTree;

public:
    HuffmanTree();
    HuffmanTree(std::unordered_map<unsigned int, std::string>& huffmanCode, HuffmanNode* MyTree);
    // 从字典创建huffman树
    void buildHuffmanTree(const std::unordered_map<unsigned int, long long>& freqMap);

    // 生成huffman编码，并返回一个用于调用的huffman树
    void generateHuffmanCodes(HuffmanNode* root, std::string code);

    // 构建huffmanTree类，从字典创建huffman树，同时生成huffman编码，将上面两个函数整合为一个
    void initializeHuffmanTree(const std::unordered_map<unsigned int, long long>& freqMap);

    // huffman编码函数，输入为一string，返回为一比特流
    void encode(const std::string& str);

    std::string bitProcess(const std::string& str);

    std::string decode(const std::string& str);

    HuffmanNode* convertFromBit(unsigned long long input, HuffmanNode* head, std::string& output, int number=-1);

    std::string bitProcessEncoder(const std::string& str);

};

#endif