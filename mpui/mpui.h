#ifndef MPUI_H
#define MPUI_H

#include <thread>

namespace mpui
{

enum MPUI_Mode { HUB=0, SOURCE };

struct MPUI_WSize
{
    int x;
    int y;
    int z;
};

struct MPUI_Session
{
    MPUI_Mode    mode;
    MPUI_WSize   wsize;
    std::thread *hubloopth;
};

void MPUI_Init      ( MPUI_Mode mode, MPUI_WSize wsize, MPUI_Session *&session );
void MPUI_Finalize  ( MPUI_Session *&session );
void MPUI_Recv_local( MPUI_Session *session, double *buff );

void MPUI_Hub_init     ( std::thread *&loopth );
void MPUI_Hub_finalize ( std::thread *&loopth );
void MPUI_Hub_setWSize ( int xsize, int ysize, int zsize );
void MPUI_Hub_setBuffer( double *buff, int xsize, int ysize, int zsize );
void MPUI_Hub_filter   ( double threshold );
bool MPUI_Hub_onexit   ();

}  // namespace mpui

#endif