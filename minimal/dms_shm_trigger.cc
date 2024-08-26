#include <iostream>
#include <fstream>
#include <thread>

int main(int argc, char** argv) {
    
    // write pingPongReady signal to file pingPongReady.bin every seconds
    int pingPongReady = 0;
    while (true) {
        std::ofstream file("pingPongReady.bin", std::ios::binary);
        file.write(reinterpret_cast<char*>(&pingPongReady), sizeof(pingPongReady));
        file.close();
        std::cout << "pingPongReady: " << pingPongReady << std::endl;
        pingPongReady = 1 - pingPongReady;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    return 0;
}