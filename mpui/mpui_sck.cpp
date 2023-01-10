#include "mpui.h"

#include <string.h>
#include <vector>

#include <stdio.h> // remove it! just for printf
#include <iostream>

#define __MPUI_PORT__ 4320

namespace mpui {

unsigned int __compress( double *buff_in, unsigned int buff_in_size, std::vector<unsigned char> &buff_bytes )
{
    unsigned int nbytes = 0;
    int i=0;
    while ( i < buff_in_size )
    {
        double val = buff_in[i];
        unsigned int n=1;
        
        while ( n < 0xFF && ++i < buff_in_size && buff_in[i] == val )
            ++n;

        if ( n >= 0xFF ) ++i;

        buff_bytes.push_back((unsigned char)n);
        ++nbytes;

        float fval = (float) val;
        const unsigned char *val_bytes = reinterpret_cast<const unsigned char*>(&fval);

        for ( int j=0; j<sizeof(float); ++j )
        {
            buff_bytes.push_back(val_bytes[j]);
            ++nbytes;
        }
    }
    return nbytes;
}

unsigned int __decompress( std::vector<unsigned char> &buff_bytes, double *buff_out )
{
    unsigned int buff_in_size = buff_bytes.size();
    unsigned int i=0;
    unsigned int j=0;
    while ( i < buff_in_size )
    {
        const unsigned char n = buff_bytes[i];
        const float val = *reinterpret_cast<float*>(&(*buff_bytes.begin()) + i + 1);

        for ( unsigned int k=0; k < n; ++k )
            buff_out[j++] = (double) val;
        
        i += 1 + sizeof(float);
    }
    return j;
}

int MPUI_Send( MPUI_Session *session, double *buff, const char *hostname, unsigned long dt )
{
    struct timespec currtime;
    clock_gettime( CLOCK_MONOTONIC, &currtime );
    
    // time diff in millis.
    unsigned long timediff = 1e3  * (currtime.tv_sec - session->lsend_time.tv_sec) +
                             1e-6 * (currtime.tv_nsec - session->lsend_time.tv_nsec);
    if ( timediff < dt ) return 0;
    session->lsend_time = currtime;
    
    if ( session->sockfd < 0 )
    {
        session->sockfd = socket(AF_INET, SOCK_STREAM, 0);
        session->hubaddr.sin_family = AF_INET;
        session->hubaddr.sin_port = htons(__MPUI_PORT__);
        inet_pton( AF_INET, hostname, &session->hubaddr.sin_addr);
        connect(session->sockfd, (struct sockaddr*)&session->hubaddr, sizeof(session->hubaddr));
    }

    std::vector<unsigned char> buff_bytes;
    for ( int i=0; i < sizeof(unsigned int); ++i ) buff_bytes.push_back(0);
    
    const unsigned int   buffsize     = session->wsize.x * session->wsize.y * session->wsize.z;
    const unsigned int   nbytes       = __compress( buff, buffsize, buff_bytes );
    const unsigned char *nbytes_bytes = reinterpret_cast<const unsigned char*>(&nbytes);
    for ( int i=0; i < sizeof(unsigned int); ++i ) buff_bytes[i] = nbytes_bytes[i];
    
    send( session->sockfd, &(*buff_bytes.begin()), buff_bytes.size(), 0 );
    
    printf("Sent\n");
    return 0;
}

int MPUI_Recv( MPUI_Session *session, double *buff )
{
    if ( session->eof ) return 0;
    if ( std::cin.eof() )
    {
        session->eof = true;
        return 0;
    }

    unsigned int recvsize; std::cin >> recvsize;
    std::vector<unsigned char> buff_bytes;
    unsigned int b;
    for ( unsigned int i=0; i < recvsize; ++i )
    {
        std::cin >> b;
        buff_bytes.push_back(b);
    }

    __decompress( buff_bytes, buff );
    printf("Received\n");

    MPUI_Recv_local( session, buff );
    return 0;
}

}