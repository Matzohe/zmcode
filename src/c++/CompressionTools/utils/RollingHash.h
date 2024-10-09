#ifndef ROLLINGHASH_H
#define ROLLINGHASH_H

#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <deque>
#include <sstream>
#include <bitset>
#include <fstream>

class RollingHash{

private:
    const int B;  // 基数，通常选择较小的素数
    const int M;  // 模数，选择较大的素数以避免哈希冲突
    const int subStrLen;  // 计算所有长度为 subStrLen 的子串的哈希值
    const long long power; // 用于滚动哈希的处理
    long long StarthashValue; // 用于滚动哈希的处理，移出窗口之外的字符串
    long long EndhashValue; // 用于添加新的输入的哈希值
    int StartPosition; // 滚动哈希开始的位置
    int CurrentPosition; // 当前扫描到的位置
    int MaxLen; // 缓冲窗口的最长长度
    std::unordered_map<int, long long> charcount; // 存储每个字节的数量
    std::unordered_map<int, std::deque<int> > hashDict;
    std::string decodeInfo;

public:

    
    RollingHash();
    RollingHash(int subStrLen);

    // 计算str前subStrLen个字符的哈希值
    int ComputeHash(const std::string& str);

    // 将新的字符串添加到哈希表中
    void AddRollingHash(const std::string& str, int current_position);

    // 将末尾的字符串从哈希表中删除
    void DropRollingHash(const std::string& str);

    // 查找哈希表
    int Find(int hashValue, const std::string& str, int CurrentPosition);

    // 初始化滚动哈希表
    void InitRollingHash(const std::string& str);

    // 找到了相同哈希的子串，寻找最大匹配子串的长度
    int FindMaxMatch(const std::string & str, int StartIndex, int CurrentPosition);

    // 将子串搭建成LZ77的格式
    std::string EncodeLZ77(int before_position, int length);

    // 使用滚动哈希实现LZ77压缩，返回压缩后的字符串
    std::string RollingHashProcess(const std::string& str);

    std::unordered_map<unsigned int, long long> getCharcount(std::string str);

    std::string DecodeLZ77(std::string str);

    std::string DecodeLZ77WithInfo(std::string str);

    std::string getDecodeInfo(){return decodeInfo;};

    void setDecodeInfo(std::string info){decodeInfo = info;};
    
};

#endif