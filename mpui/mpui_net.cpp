#include "mpui.h"

#include <string.h>
#include <vector>

#include <stdio.h> // remove it! just for printf
#include <iostream>

#define __MPUI_PORT__ 4320
#define __MPUI_MTU__  50000
#define __BUFFSIZE__       (session->wsize.x * session->wsize.y * session->wsize.z)
#define __SLICE_BUFFSIZE__ (session->wsize.x * session->wsize.y)

namespace mpui {

void __compress( double *buff_in, unsigned int buff_in_size, std::vector<unsigned char> &buff_out_vector )
{
    int i=0;
    while ( i < buff_in_size )
    {
        double val = buff_in[i];
        unsigned int n=1;
        
        while ( n < 0xFF && ++i < buff_in_size && buff_in[i] == val )
            ++n;

        if ( n >= 0xFF ) ++i;

        buff_out_vector.push_back((unsigned char)n);

        float fval = (float) val;
        const unsigned char *val_bytes = reinterpret_cast<const unsigned char*>(&fval);

        for ( int j=0; j<sizeof(float); ++j )
            buff_out_vector.push_back(val_bytes[j]);
    }
}

unsigned int __decompress( unsigned char *buff_in, unsigned int buff_in_size, double *buff_out )
{
    unsigned int i=0;
    unsigned int j=0;
    while ( i < buff_in_size )
    {
        const unsigned char n = buff_in[i];
        const float val = *reinterpret_cast<float*>(buff_in + i + 1);

        for ( unsigned int k=0; k < n; ++k )
            buff_out[j++] = (double) val;
        
        i += 1 + sizeof(float);
    }
    return j;
}

template<typename T>
void __push_bytes( std::vector<unsigned char> &bytes_vector, T val )
{
    const unsigned char *bytes = reinterpret_cast<const unsigned char*>(&val);
    for ( int k=0; k < sizeof(T); ++k ) bytes_vector.push_back( bytes[k] );
}

int __fit_and_send( MPUI_Session *session, std::vector<unsigned char> &buff_out_vector, unsigned int buff_out_offset = 0 )
{
    const unsigned buff_out_size = buff_out_vector.size();
    if ( buff_out_size + sizeof(unsigned int) + sizeof(unsigned int) <= __MPUI_MTU__ )
    {
        __push_bytes<unsigned int>( buff_out_vector, buff_out_offset );
        __push_bytes<unsigned int>( buff_out_vector, session->seqn   );
        
        sendto( session->sockfd, &(*buff_out_vector.begin()), buff_out_vector.size(), 0, 
                (struct sockaddr*)&session->hubaddr, sizeof(session->hubaddr));
        printf("Sent %ld bytes with offset %d\n", buff_out_size + sizeof(unsigned int), buff_out_offset);
        return 0;
    }

    unsigned int subbuff_out_size = buff_out_size / 2;
    while ( subbuff_out_size % (1 + sizeof(float)) != 0 ) --subbuff_out_size;

    std::vector<unsigned char> *subbuff_out = new std::vector<unsigned char>
        (buff_out_vector.begin(), buff_out_vector.begin() + subbuff_out_size);
    
    unsigned int left_subbuff_actual_size = 0;
    for ( int k=0; k < subbuff_out->size(); k += 1 + sizeof(float) )
        left_subbuff_actual_size += (*subbuff_out)[k];
    
    int ans = __fit_and_send( session, *subbuff_out, buff_out_offset  );
    delete subbuff_out;
    
    if ( ans != 0 ) return ans;

    subbuff_out = new std::vector<unsigned char>
        (buff_out_vector.begin() + subbuff_out_size, buff_out_vector.end());
    
    ans = __fit_and_send( session, *subbuff_out, buff_out_offset + left_subbuff_actual_size );
    delete subbuff_out;

    return ans;
}

int MPUI_Send( MPUI_Session *session, double *buff, const char *hostname )
{
    if ( session->sockfd < 0 )
    {
        session->sockfd = socket(PF_INET, SOCK_DGRAM, 0);
        memset(&session->hubaddr, '\0', sizeof(session->hubaddr));
        session->hubaddr.sin_family = AF_INET;
        session->hubaddr.sin_port = htons(__MPUI_PORT__);
        session->hubaddr.sin_addr.s_addr = inet_addr(hostname);
    }

    std::vector<unsigned char> buff_out_vector;
    __compress( buff, __BUFFSIZE__, buff_out_vector );

    int ans = __fit_and_send( session, buff_out_vector );
    ++(session->seqn);
    
    return ans;
}

int MPUI_Recv( MPUI_Session *session, double *buff )
{
    if ( session->sockfd < 0 )
    {
        session->sockfd = socket(AF_INET, SOCK_DGRAM, 0);
        memset(&session->hubaddr, 0, sizeof(session->hubaddr));
        session->hubaddr.sin_family = AF_INET;
        session->hubaddr.sin_port = htons(__MPUI_PORT__);
        session->hubaddr.sin_addr.s_addr = INADDR_ANY;
        bind(session->sockfd, (struct sockaddr*)&session->hubaddr, sizeof(session->hubaddr));
    }

    unsigned char recv_buff[__MPUI_MTU__];
    struct sockaddr_in sourceaddr;
    socklen_t addr_size = sizeof(sourceaddr);

    unsigned int total_received_size = 0;
    do
    {
        ssize_t recv_size = recvfrom(session->sockfd, recv_buff, __MPUI_MTU__, 0, (struct sockaddr*)& sourceaddr, &addr_size);
        printf("Received %ld bytes\n", recv_size);

        unsigned int recv_offset = *reinterpret_cast<unsigned int*>(recv_buff + recv_size - sizeof(unsigned int) * 2);
        unsigned int recv_seqn   = *reinterpret_cast<unsigned int*>(recv_buff + recv_size - sizeof(unsigned int));

        if ( recv_seqn < session->seqn ) continue;
        if ( recv_seqn > session->seqn )
        {
            session->seqn = recv_seqn;
            total_received_size = 0;
        }

        total_received_size += __decompress( recv_buff, recv_size - sizeof(unsigned int), buff + recv_offset );
        MPUI_Recv_local( session, buff );
    }
    while( total_received_size < __BUFFSIZE__ );

    return 0;
}

}