# DV2606 - Lab2 GPU programming

Written by:
* Emil Karlström
* Samuel Jonsson 

# Featured new flags

* `-t` controlls the amount of threads to use for each dimension of a 2D block of threads. The total amount of threads per block is the square of this number, i.e `-t 4` would use a 4x4 block size, that is, 16 threads. 

* `-v` controlls whether or not to perform verification. This uses the gauss-jordan sequential algorithm from Grama (Algorithm 8.4) to compute the correct answer, then compares the answers to 10 decimal places. The reason we choose 10 decimals is that we consider that being enough accuracy whilst still having some spare room in case of floating point precision rounding errors were to occur. 