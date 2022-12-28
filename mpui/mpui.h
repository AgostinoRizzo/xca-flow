#ifndef MPUI_H
#define MPUI_H

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
    MPUI_Mode  mode;
    MPUI_WSize wsize;
};

void MPUI_Init( MPUI_Mode mode, MPUI_WSize wsize, MPUI_Session *&session );
void MPUI_Finalize( MPUI_Session *&session );
void MPUI_Recv_local( MPUI_Session *session, double *buff );

int  MPUI_Hub_init     ();
void MPUI_Hub_setBuffer( double *buff, int xsize, int ysize, int zsize );
void MPUI_Hub_mainloop ();
void MPUI_Hub_filter   ( double threshold );

}  // namespace mpui

#endif