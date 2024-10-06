#include "HuffmanTree.h"

struct Compare {
    bool operator()(HuffmanNode* l, HuffmanNode* r) {
        return l->freq > r->freq;
    }
};

HuffmanTree::HuffmanTree(std::unordered_map<unsigned int, std::string>& huffmanCode, HuffmanNode* MyTree): huffmanCode(huffmanCode), MyTree(MyTree) {
}

void HuffmanTree::initializeHuffmanTree(const std::unordered_map<unsigned int, long long>& freqMap) {
    buildHuffmanTree(freqMap);
    generateHuffmanCodes(MyTree, "");
}


void HuffmanTree::buildHuffmanTree(const std::unordered_map<unsigned int, long long>& freqMap) {

    std::priority_queue<HuffmanNode*, std::vector<HuffmanNode*>, Compare> minHeap;
    for (const auto& pair : freqMap) {
        minHeap.push(new HuffmanNode(pair.first, pair.second));
    }

    while (minHeap.size() != 1) {
        HuffmanNode* left = minHeap.top();
        minHeap.pop();
        HuffmanNode* right = minHeap.top();
        minHeap.pop();

        HuffmanNode* sum = new HuffmanNode('\0', left->freq + right->freq);
        sum->left = left;
        sum->right = right;
        minHeap.push(sum);
    }

    MyTree = minHeap.top();
}

void HuffmanTree::generateHuffmanCodes(HuffmanNode* root, std::string code) {
    if (!root) return;

    // 如果是叶子节点，保存字符及其编码
    if (!root->left && !root->right) {
        huffmanCode[root->ch] = code;
    }

    // 递归处理左右子树
    generateHuffmanCodes(root->left, code + "0");
    generateHuffmanCodes(root->right, code + "1");
}

// TODO:Encode部分可以进行多线程
void HuffmanTree::encode(const std::string& str) {
    std::string result = "";
    std::string code = "";
    for (int i = 0; i < str.size(); i++) {
        unsigned int charCode = static_cast<unsigned int>(str[i]);
        result.append(huffmanCode[charCode]);
        if (result.size() >= 320){
            code.append(bitProcessEncoder(result.substr(0, 320)));
            std::string new_result = result.substr(320, result.size() - 320);
            result = new_result;
        }
    }
    if (result.size() > 0){
        int all_number = result.size() / 8;
        for (int i = 0; i < all_number; i++){
            std::string temp = result.substr(i * 8, 8);
            std::reverse(temp.begin(), temp.end());
            std::bitset<8> bitset(temp);
            unsigned long num = bitset.to_ulong();
            unsigned char c = static_cast<unsigned char>(num);
            code.push_back(c);
        }
        for (int i = 0; i < result.size() % 8; i++){
            code.push_back(result[all_number * 8 + i]);
        }
        // 在末尾添加标注，表明有多少个不完整的字符
        code.push_back(char(result.size() % 8 + '0'));
    }

    std::ofstream current_file("new_example.txt", std::ios::app);
    current_file << code;
    current_file.close();
}

// TODO: bit process也可以进行多线程加速
std::string HuffmanTree::bitProcess(const std::string& str){
    std::string output = "";
    for (int i = 0; i < 5; i++){
        std::string tp = str.substr(i * 64, 64);
        std::bitset<64> bitset(tp);
        std::string temp(8, '\0');
        unsigned long long value = bitset.to_ullong();
        std::memcpy(&temp, &value, sizeof(value));
        output.append(temp);
    }
    return output;
}

std::string HuffmanTree::bitProcessEncoder(const std::string& str){
    std::string output = "";
    for (int i = 0; i < 5; i++){
        std::string tp = str.substr(i * 64, 64);
        std::reverse(tp.begin(), tp.end());
        std::bitset<64> bitset(tp);
        std::string temp(8, '\0');
        unsigned long long value = bitset.to_ullong();
        std::memcpy(&temp, &value, sizeof(value));
        output.append(temp);
    }
    return output;
}


std::string HuffmanTree::decode(const std::string& input_str) {
    // 将比特序列转化为字符串
    std::string output = "";
    int leave_number = input_str[input_str.size() - 1] - '0';

    std::string str = input_str.substr(0, input_str.size() - leave_number - 1);

    int char_length = str.size() / 8;
    HuffmanNode* head = MyTree;
    for (int i = 0; i < char_length; i++){
        unsigned long long input;
        std::memcpy(&input, &str[i * sizeof(input)], sizeof(input));
        head = convertFromBit(input, head, output, -1);

    }
    if (char_length * 8 < str.size()){
        unsigned long long input = 0;
        std::memcpy(&input, &str[char_length * 8], str.size() - char_length * 8);
        head = convertFromBit(input, head, output, (str.size() - char_length * 8) * 8);
    }
    for (int i = 0; i < leave_number; i++){
        if (input_str[i + str.size()] == '0'){
            head = head->left;
            if (!head->left && !head->right){
                unsigned char ch = head->ch;
                output.push_back(ch);
                head = MyTree;
            }
        }
        else{
            head = head->right;
            if (!head->left && !head->right){
                unsigned char ch = head->ch;
                output.push_back(ch);
                head = MyTree;
            }
        }
    }
    return output;
}


HuffmanNode* HuffmanTree::convertFromBit(unsigned long long input, HuffmanNode* head, std::string& output, int number) {
    int all_num;
    if (number != -1){
        all_num = number;
    }
    else{
        all_num = 64;
    }
    for (int i = 0; i < all_num; i++){
        if ((input >> i) & 1){
            head = head->right;
            if (!head->left && !head->right){
                output.push_back(head->ch);
                head = MyTree;
            }
        }
        else{
            head = head->left;
            if (!head->left && !head->right){
                output.push_back(head->ch);
                head = MyTree;
            }
        }
    }
    return head;
}