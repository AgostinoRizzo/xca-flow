#include "mpui.h"

namespace mpui {

void
MPUI_Init( MPUI_Mode mode, MPUI_WSize wsize, MPUI_Session *&session )
{
    session = new mpui::MPUI_Session;
    session->mode = mode;
    session->wsize = wsize;
    session->hubloopth = nullptr;

    if ( mode == MPUI_Mode::HUB )
    {
        MPUI_Hub_init( session->hubloopth );
        MPUI_Hub_filter( -734.0f );
        MPUI_Hub_setWSize( session->wsize.x, session->wsize.y, session->wsize.z );
    }
}

void
MPUI_Finalize( MPUI_Session *&session )
{
    if ( session->mode == MPUI_Mode::HUB )
        MPUI_Hub_finalize( session->hubloopth );
    delete session;
    session = nullptr;
}

void
MPUI_Recv_local( MPUI_Session *session, double *buff )
{
    MPUI_Hub_setBuffer(buff, session->wsize.x, session->wsize.y, session->wsize.z);
}

}