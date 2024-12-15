#include <fstream>
#include <iostream>

#include "apis_c.h"

#define width 1

int idX, idY;
// 本程序旨在计算一个（4*width,width)与(width,4*width)的矩阵相乘
int main(int argc, char** argv) {
    idX = atoi(argv[1]);
    idY = atoi(argv[2]);

    if (idX == 0 && idY == 0) {
        int64_t* A = (int64_t*)malloc(sizeof(int64_t) * width * width * 4);
        int64_t* B = (int64_t*)malloc(sizeof(int64_t) * width * width * 4);
        int64_t* C1 = (int64_t*)malloc(sizeof(int64_t) * width * width * 16);
        int64_t* C2 = C1 + width * width * 4;
        int64_t* C3 = C1 + width * width * 8;
        int64_t* C4 = C1 + width * width * 12;

        // for (int i = 0; i < width * width * 4; i++)
        // {
        //     A[i] = rand() % 51;
        //     B[i] = rand() % 51;
        // }
        InterChiplet::receiveMessage(idX, idY, 0, 1, A, 1);

        InterChiplet::receiveMessage(idX, idY, 0, 1, A,
                                     4 * width * width * sizeof(int64_t));
        InterChiplet::receiveMessage(idX, idY, 0, 2, B,
                                     4 * width * width * sizeof(int64_t));

        InterChiplet::sendMessage(1, 0, idX, idY, A,
                                  width * width * sizeof(int64_t));
        InterChiplet::sendMessage(2, 0, idX, idY, A + width * width,
                                  width * width * sizeof(int64_t));
        InterChiplet::sendMessage(3, 0, idX, idY, A + width * width * 2,
                                  width * width * sizeof(int64_t));
        InterChiplet::sendMessage(4, 0, idX, idY, A + width * width * 3,
                                  width * width * sizeof(int64_t));

        InterChiplet::sendMessage(1, 0, idX, idY, B,
                                  4 * width * width * sizeof(int64_t));
        InterChiplet::sendMessage(2, 0, idX, idY, B,
                                  4 * width * width * sizeof(int64_t));
        InterChiplet::sendMessage(3, 0, idX, idY, B,
                                  4 * width * width * sizeof(int64_t));
        InterChiplet::sendMessage(4, 0, idX, idY, B,
                                  4 * width * width * sizeof(int64_t));

        InterChiplet::receiveMessage(idX, idY, 1, 0, C1,
                                     4 * width * width * sizeof(int64_t));
        InterChiplet::receiveMessage(idX, idY, 2, 0, C2,
                                     4 * width * width * sizeof(int64_t));
        InterChiplet::receiveMessage(idX, idY, 3, 0, C3,
                                     4 * width * width * sizeof(int64_t));
        InterChiplet::receiveMessage(idX, idY, 4, 0, C4,
                                     4 * width * width * sizeof(int64_t));
    } else if (idX == 0 && idY == 1) {

        int64_t* A = (int64_t*)malloc(sizeof(int64_t) * width * width * 4);
        InterChiplet::sendMessage(0, 0, 0, 1, A, 1);
        for (int i = 0; i < width * width * 4; i++) {
            A[i] = rand() % 51;
        }
        InterChiplet::sendMessage(0, 0, 0, 1, A,
                                  4 * width * width * sizeof(int64_t));

        return 0;
    } else if (idX == 0 && idY == 2) {
        int64_t* B = (int64_t*)malloc(sizeof(int64_t) * width * width * 4);
        for (int i = 0; i < width * width * 4; i++) {
            B[i] = rand() % 51;
        }
        InterChiplet::sendMessage(0, 0, 0, 2, B,
                                  4 * width * width * sizeof(int64_t));

        return 0;
    }
}
