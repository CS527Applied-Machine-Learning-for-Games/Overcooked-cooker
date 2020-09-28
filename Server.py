import socket
import time
import sys
import csv

HOST = ''
PORT = 5555
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen(1)
    conn, addr = s.accept()
    with open("./data/data_0_gamedata.csv","w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        with conn:
            while True:
                conn.sendall(str.encode("request"))
                data = conn.recv(1024)
                if data:
                    sys.stdout.write("\r%s    " %(bytes.decode(data)))
                    sys.stdout.flush()
                    writer.writerow([bytes.decode(data)])
                time.sleep(0.05)