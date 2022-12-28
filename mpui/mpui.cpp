#include "mpui.h"

namespace mpui {

void
MPUI_Init( MPUI_Mode mode, MPUI_WSize wsize, MPUI_Session *&session )
{
    session = new mpui::MPUI_Session;
    session->mode = mode;
    session->wsize = wsize;

    if ( mode == MPUI_Mode::HUB )
    {
        MPUI_Hub_init();
        MPUI_Hub_filter( -734.0f );
    }
}

void
MPUI_Finalize( MPUI_Session *&session )
{
    delete session;
    session = nullptr;
}

void
MPUI_Recv_local( MPUI_Session *session, double *buff )
{
    MPUI_Hub_setBuffer(buff, session->wsize.x, session->wsize.y, session->wsize.z);
    MPUI_Hub_mainloop();
}

}