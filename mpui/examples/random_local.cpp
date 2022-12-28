#include "../mpui.h"
#include <stdlib.h>
#include <chrono>
#include <thread>

#define WSIZE 10

int main()
{
    mpui::MPUI_WSize wsize;
    wsize.x = WSIZE;
    wsize.y = WSIZE;
    wsize.z = WSIZE;
    
    mpui::MPUI_Session *session;
    mpui::MPUI_Init(mpui::MPUI_Mode::HUB, wsize, session);
    mpui::MPUI_Hub_filter(0.4f);

    int size = wsize.x * wsize.y * wsize.z;
    double *buffer = new double[size];

    srand(time(nullptr));

    while ( !mpui::MPUI_Hub_onexit() )
    {
        for ( int i=0; i<size; ++i )
            buffer[i] = (double) rand() / RAND_MAX;
        
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        mpui::MPUI_Recv_local(session, buffer);
    }
    
    mpui::MPUI_Finalize(session);
    delete[] buffer;
    return 0;
}