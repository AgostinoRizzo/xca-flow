#define __MPUI_HUB__
#include <string>
#include "../mpui.h"

#define SET3D(M, rows, columns, i, j, k, value) \
    ( (M)[( ((k)*(rows)*(columns)) + ((i)*(columns)) + (j) )] = (value) )

void readBuffer(double* buff, int r, int c, int s, std::string path)
{
    FILE *f = fopen(path.c_str(), "r");
    if (f == NULL)
    {
        printf("can not open file %s", path.c_str());
        exit(0);
    }
    
    char str[256];
    int i, j, k;
    for (k = 0; k < s; k++) {
    for (i = 0; i < r; i++)
    {
        for (j = 0; j < c; j++)
        {
            fscanf(f, "%s", str); fscanf(f, " ");
            SET3D(buff, r, c, i, j, k, atof(str));
        }
        fscanf(f, "\n");
    }
    fscanf(f, "\n"); }
    fclose(f);
}

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

    double *buffer = new double[wsize.x * wsize.y * wsize.z];
    readBuffer( buffer, wsize.y, wsize.x, wsize.z, "output_h_LAST_simulation_time_864000s.txt" );
    mpui::MPUI_Recv_local(session, buffer);
    
    mpui::MPUI_Finalize(session);
    delete[] buffer;

    return 0;
}