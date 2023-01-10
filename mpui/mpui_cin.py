import socket

__MPUI_PORT__ = 4320

try:
    with socket.socket( socket.AF_INET, socket.SOCK_STREAM ) as s:
        s.bind(('0.0.0.0', __MPUI_PORT__))
        s.listen()
        conn, addr = s.accept()

        with conn:
            while True:
                recv_bytes = conn.recv(4)
                if not recv_bytes: break
                n = int.from_bytes(recv_bytes, byteorder="little", signed=False)
                print(" %d" % n, end='')
                for _ in range(n):
                    print(" %d" % int.from_bytes(conn.recv(1), byteorder="little", signed=False), end='')
except KeyboardInterrupt:
    pass