#include <iostream>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>
#include <fcntl.h>
#include <cstring>
#include <string>
#include <cstdlib>

// 函数声明
void copyFile(const std::string& srcFile, const std::string& destFile);
void copyDirectory(const std::string& srcDir, const std::string& destDir);

void copyFile(const std::string& srcFile, const std::string& destFile) {
    // 打开源文件
    int srcFd = open(srcFile.c_str(), O_RDONLY);
    if (srcFd < 0) {
        perror(("Failed to open source file: " + srcFile).c_str());
        exit(EXIT_FAILURE);
    }

    // 获取源文件的权限信息
    struct stat srcStat;
    if (fstat(srcFd, &srcStat) < 0) { // 使用 fstat 获取打开文件的元数据
        perror(("Failed to get source file status: " + srcFile).c_str());
        close(srcFd);
        exit(EXIT_FAILURE);
    }

    // 创建目标文件
    int destFd = open(destFile.c_str(), O_WRONLY | O_CREAT | O_TRUNC, srcStat.st_mode);
    if (destFd < 0) {
        perror(("Failed to create destination file: " + destFile).c_str());
        close(srcFd);
        exit(EXIT_FAILURE);
    }

    // 复制文件内容
    char buffer[4096];
    ssize_t bytesRead;
    while ((bytesRead = read(srcFd, buffer, sizeof(buffer))) > 0) {
        if (write(destFd, buffer, bytesRead) != bytesRead) {
            perror("Write error");
            close(srcFd);
            close(destFd);
            exit(EXIT_FAILURE);
        }
    }

    if (bytesRead < 0) {
        perror("Read error");
    }

    // 设置目标文件权限与源文件相同
    if (fchmod(destFd, srcStat.st_mode) < 0) { // 使用 fchmod 设置目标文件权限
        perror(("Failed to set file permissions for: " + destFile).c_str());
    }

    // 关闭文件描述符
    close(srcFd);
    close(destFd);
}

void copyDirectory(const std::string& srcDir, const std::string& destDir) {
    // 获取源目录的权限
    struct stat srcStat;
    if (stat(srcDir.c_str(), &srcStat) < 0) {
        perror(("Failed to get source directory status: " + srcDir).c_str());
        exit(EXIT_FAILURE);
    }

    // 创建目标目录，权限设置为源目录的权限
    if (mkdir(destDir.c_str(), srcStat.st_mode & 0777) < 0) {
        if (errno != EEXIST) {  // 忽略目标目录已存在的错误
            perror(("Failed to create destination directory: " + destDir).c_str());
            exit(EXIT_FAILURE);
        }
    } else {
        // 如果目录新创建，明确设置权限
        if (chmod(destDir.c_str(), srcStat.st_mode & 0777) < 0) {
            perror(("Failed to set permissions for directory: " + destDir).c_str());
        }
    }

    // 打开源目录
    DIR* dir = opendir(srcDir.c_str());
    if (!dir) {
        perror(("Failed to open source directory: " + srcDir).c_str());
        exit(EXIT_FAILURE);
    }

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        // 跳过 "." 和 ".."
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }

        // 构造源路径和目标路径
        std::string srcPath = srcDir + "/" + entry->d_name;
        std::string destPath = destDir + "/" + entry->d_name;

        // 获取文件或目录的元信息
        struct stat entryStat;
        if (stat(srcPath.c_str(), &entryStat) < 0) {
            perror(("Failed to stat file: " + srcPath).c_str());
            closedir(dir);
            exit(EXIT_FAILURE);
        }

        if (S_ISDIR(entryStat.st_mode)) {
            // 如果是目录，递归复制
            copyDirectory(srcPath, destPath);
        } else if (S_ISREG(entryStat.st_mode)) {
            // 如果是文件，复制文件
            copyFile(srcPath, destPath);
        }
    }

    closedir(dir);
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <source> <destination>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string src = argv[1];
    std::string dest = argv[2];

    struct stat srcStat;
    if (stat(src.c_str(), &srcStat) < 0) {
        perror(("Failed to stat source: " + src).c_str());
        return EXIT_FAILURE;
    }

    if (S_ISDIR(srcStat.st_mode)) {
        // 如果是目录，调用目录复制函数
        copyDirectory(src, dest);
    } else if (S_ISREG(srcStat.st_mode)) {
        // 如果是文件，调用文件复制函数
        copyFile(src, dest);
    } else {
        std::cerr << "Unsupported file type: " << src << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
