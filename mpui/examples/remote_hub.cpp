#define __MPUI_HUB__
#include "../mpui.h"

int main()
{
    mpui::MPUI_WSize wsize;
    wsize.x = 100;
    wsize.y = 100;
    wsize.z = 50;
    
    mpui::MPUI_Session *session;
    mpui::MPUI_Init(mpui::MPUI_Mode::HUB, wsize, session);
    mpui::MPUI_Hub_setRange( -730.0, 5600.0 );
    mpui::MPUI_Hub_filter( -734.0f );

    const unsigned int buffsize = wsize.x * wsize.y * wsize.z;
    double *buffer = new double[buffsize];
    for ( int i=0; i<buffsize; ++i ) buffer[i] = -734.0f;
    
    while ( !mpui::MPUI_Flag_EOF(session) )
        mpui::MPUI_Recv(session, buffer);
    
    mpui::MPUI_Finalize(session);
    delete[] buffer;

    return 0;
}