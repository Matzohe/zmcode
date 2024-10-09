#include "RollingHash.h"


RollingHash::RollingHash(): B(257), M(1e9 + 9), subStrLen(3), power(257 * 257), EndhashValue(0), StarthashValue(0), hashDict(), StartPosition(0), CurrentPosition(0), MaxLen(32 * 1024), decodeInfo("") {
    
}

RollingHash::RollingHash(int subStrLen): B(257), M(1e9 + 9), subStrLen(subStrLen), power(257 * 257), EndhashValue(0), StarthashValue(0), hashDict(), StartPosition(0), CurrentPosition(0), MaxLen(32 * 1024), decodeInfo("") {
    
}


// 计算给定字符串的前subStrLen个字符串的哈希值
int RollingHash::ComputeHash(const std::string& str) {
    int hashValue_hidden = 0;
    for (int i = 0; i < subStrLen; i++) {
        hashValue_hidden = (hashValue_hidden * B + (str[i] & 0x000000FF)) % M;
    }
    return hashValue_hidden;
}

// 在保存的字典中，前面是当前字节的哈希值，后面是该哈希值对应的子串的起始位置
// 这里传入的是需要添加的字符串，需要包含传入字符串前3个字符，这意味着传入的字符串大小为 3 + str。
// current position是指在当原字符串的位置
// 这个函数的作用是向哈希表里面添加新增的字符串

void RollingHash::AddRollingHash(const std::string& str, int CurrentPosition) {
    for (int i = 0; i < str.size() - subStrLen + 1; i++){
         // 加上新字符，同时更新字符串字典
        unsigned char newChar = str[i + subStrLen - 1];
        charcount[newChar] += 1;
        int number = ComputeHash(str.substr(i, subStrLen));
        hashDict[number].push_back(CurrentPosition + i - subStrLen + 1);
    }
    return;
}

void RollingHash::DropRollingHash(const std::string& str){
    for (int i = 0; i < str.size() - subStrLen + 1; i++){
        int number = ComputeHash(str.substr(i, subStrLen));
        hashDict[number].pop_front();
    }
}

int RollingHash::Find(int hashValue, const std::string& str, int CurrentPosition){
    // 如果没有找到，则返回一个-1
    // 如果找到了，返回最近子串的起始位置
    if (hashDict[hashValue].empty()){
        hashDict.erase(hashValue);
        return -1;
    }
    else{
        int max_index = 0;
        int max_length = -2;
        for (int i = 0; i < hashDict[hashValue].size(); i++){
            int length = FindMaxMatch(str, hashDict[hashValue][i], CurrentPosition);
            if (length > max_length){
                max_index = i;
                max_length = length;
            }
        }
        if (max_length == 0){
            return -1;
        }
        return hashDict[hashValue][max_index];
    } 
}


void RollingHash::InitRollingHash(const std::string& str){
    int number = ComputeHash(str.substr(0, subStrLen));
    hashDict[number].push_back(0);
    StartPosition = 0;
    CurrentPosition = subStrLen;
    for (int i = 0; i < subStrLen; i++){
        charcount[str[i]] += 1;
    }
    decodeInfo.append("000");
}

int RollingHash::FindMaxMatch(const std::string & str, int current_index, int CurrentPosition){
    int length = subStrLen;

    if (str[current_index] != str[CurrentPosition] || str[current_index + 1] != str[CurrentPosition + 1] || str[current_index + 2] != str[CurrentPosition + 2]){
        return 0;
    }

    while(current_index + length < CurrentPosition && CurrentPosition + length < str.size() && length <= 255){
        if (str[current_index + length] == str[CurrentPosition + length]){
            length += 1;
        }
        else{
            break;
        }
    }
    return length;
}

std::string RollingHash::EncodeLZ77(int before_position, int length){
    std::string mid_str;
    unsigned char position1 = (before_position >> 8) & 0xFF;
    mid_str.push_back(position1);
    unsigned char position2 = before_position & 0xFF;
    mid_str.push_back(position2);
    unsigned char lth = length & 0xFF;
    mid_str.push_back(lth);
    return mid_str;
}


// TODO:编码部分可以进行多线程，通过创建多个文件进行多线程编码
std::string RollingHash::RollingHashProcess(const std::string& str_input){
    // 首先初始化哈希表
    InitRollingHash(str_input);
    std::string encodedString = "";
    encodedString.append(str_input.substr(0, subStrLen));

    while (CurrentPosition + subStrLen < str_input.size()){
        if (CurrentPosition - StartPosition > MaxLen + MaxLen / 2){
            DropRollingHash(str_input.substr(StartPosition, CurrentPosition - MaxLen - StartPosition + subStrLen - 1));
            StartPosition = CurrentPosition - MaxLen;
        }

        int number = ComputeHash(str_input.substr(CurrentPosition, subStrLen));
        int index = Find(number, str_input, CurrentPosition);
        
        if (index == -1){
            AddRollingHash(str_input.substr(CurrentPosition - subStrLen + 1, subStrLen), CurrentPosition);
            encodedString.push_back(str_input[CurrentPosition]);
            CurrentPosition += 1;
            decodeInfo.append("0");
        }
        else{
            int length = FindMaxMatch(str_input, index, CurrentPosition);
            int before_position = CurrentPosition - index;
            std::string info = EncodeLZ77(before_position, length);
            decodeInfo.append("1");
            AddRollingHash(str_input.substr(CurrentPosition - subStrLen + 1, subStrLen + length - 1), CurrentPosition);
            encodedString.append(info);
            CurrentPosition += length;
        }
    }
    encodedString.append(str_input.substr(CurrentPosition, str_input.size() - CurrentPosition));
    std::ofstream current_file("example.txt", std::ios::app);
    current_file << encodedString;
    current_file.close();
    return encodedString;
}

std::unordered_map<unsigned int, long long> RollingHash::getCharcount(std::string str){
    std::unordered_map<unsigned int, long long> char_number;
    for (int i = 0; i < str.size(); i++){
        unsigned char num = str[i] & 0xFF;
        char_number[num] += 1;
    }
    return char_number;
}

std::string RollingHash::DecodeLZ77(std::string str){
    std::string output = "";
    unsigned int place = 0;
    unsigned int length = 0;
    for (int i = 0; i < str.size(); i++){
        if ((str[i] == '.') && str.size() > i + 4 && (str[i + 4] == '.')){
            unsigned char place1 = str[i + 1] & 0xFF;
            unsigned char place2 = str[i + 2] & 0xFF;
            unsigned char place3 = str[i + 3] & 0xFF;
            place = place + place1;
            place = place << 8;
            place = place + place2;
            length = length + place3;
            std::string info = output.substr(output.size() - place, length);
            output.append(info);
            i += 4;
            place = 0;
            length = 0;
        }
        else{
            output.push_back(str[i]);
        }
    }
    return output;

}

std::string RollingHash::DecodeLZ77WithInfo(std::string str){
    std::string output = "";
    unsigned int place = 0;
    unsigned int length = 0;
    int current_place = 0;
    for (int i = 0; i < decodeInfo.size(); i++){
        if (decodeInfo[i] == '1'){
            unsigned char place1 = str[current_place] & 0xFF;
            unsigned char place2 = str[current_place + 1] & 0xFF;
            unsigned char place3 = str[current_place + 2] & 0xFF;
            place = place + place1;
            place = place << 8;
            place = place + place2;
            length = length + place3;
            std::string info = output.substr(output.size() - place, length);
            output.append(info);
            place = 0;
            length = 0;
            current_place += 3;
        }
        else{
            output.push_back(str[current_place]);
            current_place += 1;
        }
    }
    return output;
}