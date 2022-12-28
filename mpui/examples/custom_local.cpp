#include "../mpui.h"

int main()
{
    mpui::MPUI_WSize wsize;
    wsize.x = 3;
    wsize.y = 3;
    wsize.z = 3;
    
    mpui::MPUI_Session *session;
    mpui::MPUI_Init(mpui::MPUI_Mode::HUB, wsize, session);

    double buffer[] =
    {
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        -734.0f, -734.0f, 1.0f, 1.0f, -734.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        -734.0f, -734.0f, -734.0f, -734.0f, -734.0f, -734.0f, -734.0f, -734.0f, 1.0f
    };
    mpui::MPUI_Recv_local(session, buffer);
    
    mpui::MPUI_Finalize(session);
    return 0;
}