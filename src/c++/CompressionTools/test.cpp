#include <iostream>
#include <bitset>


int main(){
    std::string str = "01010010101000101001000100101010010101001010101";
    std::bitset<64> bitset(str);
    std::cout << bitset << std::endl;
    std::string temp(8, '\0');
    unsigned long long value = bitset.to_ullong();
    std::memcpy(&temp, &value, sizeof(value));
    std::cout << temp << std::endl;

    long long number;
    std::string s = "abcdefgh";
    std::memcpy(&number, &s, sizeof(number));
    std::cout << number << std::endl;


}