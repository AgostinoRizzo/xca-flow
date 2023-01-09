#ifndef MPUI_H
#define MPUI_H

#include <thread>
#include <arpa/inet.h>

namespace mpui
{

#define MPUIERR_SOCKET 5

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

    int sockfd;
    struct sockaddr_in hubaddr;
    unsigned int seqn;
};

void MPUI_Init      ( MPUI_Mode mode, MPUI_WSize wsize, MPUI_Session *&session );
void MPUI_Finalize  ( MPUI_Session *&session );

int  MPUI_Send( MPUI_Session *session, double *buff, const char *hostname );
int  MPUI_Recv( MPUI_Session *session, double *buff );
void MPUI_Recv_local( MPUI_Session *session, double *buff );

#ifdef __MPUI_HUB__
void MPUI_Hub_init     ( std::thread *&loopth );
void MPUI_Hub_finalize ( std::thread *&loopth );
void MPUI_Hub_setWSize ( int xsize, int ysize, int zsize );
void MPUI_Hub_setBuffer( double *buff, int xsize, int ysize, int zsize );
void MPUI_Hub_filter   ( double threshold );
bool MPUI_Hub_onexit   ();
#endif

}  // namespace mpui

#endif