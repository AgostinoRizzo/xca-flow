#include "mpui.h"
#include <unistd.h>

namespace mpui {

void
MPUI_Init( MPUI_Mode mode, MPUI_WSize wsize, MPUI_Session *&session )
{
    session = new mpui::MPUI_Session;
    session->mode = mode;
    session->wsize = wsize;
    session->hubloopth = nullptr;
    session->sockfd = -1;
    session->seqn = 0;
    session->lsend_time.tv_sec = 0;
    session->lsend_time.tv_nsec = 0;
    session->eof = false;

    #ifdef __MPUI_HUB__
    if ( mode == MPUI_Mode::HUB )
    {
        MPUI_Hub_init( session->hubloopth );
        MPUI_Hub_setWSize( session->wsize.x, session->wsize.y, session->wsize.z );
    }
    #endif
}

void
MPUI_Finalize( MPUI_Session *&session )
{
    #ifdef __MPUI_HUB__
    if ( session->mode == MPUI_Mode::HUB )
        MPUI_Hub_finalize( session->hubloopth );
    #endif
    
    if ( session->sockfd >= 0 ) close(session->sockfd);
    
    delete session;
    session = nullptr;
}

void
MPUI_Recv_local( MPUI_Session *session, double *buff )
{
    #ifdef __MPUI_HUB__
    MPUI_Hub_setBuffer(buff, session->wsize.x, session->wsize.y, session->wsize.z);
    #endif
}

bool
MPUI_Flag_EOF( MPUI_Session *session )
{
    return session->eof;
}

}