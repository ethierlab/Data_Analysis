import tdt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv



RZ_IP = '10.1.0.100'

udp = tdt.TDTUDP(host = RZ_IP, send_type=np.float32, sort_codes=2, bits_per_bin=4)
syn = tdt.SynapseAPI()

    
c = 0
fr_matrix = []

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# sc = ax.scatter([], [], [], c='b', marker='o')
# trajectory, = ax.plot([], [], [], color='r', linewidth=1, alpha=0.5)


# ax.set_xlabel('PCA 1')
# ax.set_ylabel('PCA 2')
# ax.set_zlabel('PCA 3')
# ax.set_title('Animated 3D Trajectory of PCA Neural Data')


# def init():
#     sc._offsets3d = [], [], []
#     trajectory.set_data([], [])
#     trajectory.set_3d_properties([])
#     return sc, trajectory

# def update(frame):
#     start_frame = max(0, frame - 9)  # Update every 10 points
#     sc._offsets3d = pca_data[start_frame:frame + 1, 0], pca_data[start_frame:frame + 1, 1], pca_data[start_frame:frame + 1, 2]
#     trajectory.set_data(pca_data[start_frame:frame + 1, 0], pca_data[start_frame:frame + 1, 1])
#     trajectory.set_3d_properties(pca_data[start_frame:frame + 1, 2])
#     return sc, trajectory

fr_export = []

# avg firing rate for all channels
try:
    while syn.getMode() == 3 or syn.getMode() == 2 :
        data = udp.recv()
        sort_code = 1
        sc = data[sort_code-1]
        fr = []
        for ch in (1,2,3,5,7,13,14,15,17,19,22, 23,24,26,28,30,32):
        # for ch in range(1, 33):
        # channel = 10
            fr.append(sc[ch-1])    
            #append to the fr by channel by time bins:
        fr_export.append(fr)
        # print(fr_export)
        # shape of fr_export
        #x    | ch1 | ch2 |ch3 | ...
        # -------------------------
        #bin1 | x   |     |    |
        # ... |     |     |    |
        # afr = np.average(fr)

        # updating every 10 time bins
        # if c == 10:
        # # PCA
        #     pca = PCA(n_components=3)

        #     test = pca.fit_transform(fr_matrix)
        #     scalar = MinMaxScaler(feature_range=(0,1))
        #     X = scalar.fit_transform(test)

        #     # X = test.reshape(-1, 1)
        #     # print(X)

        #     # print('average firing rate: ', afr, end='\t\t\t\r')

        #     # print('CHANNEL:', channel, 'SORT:', sort_code, '\t', sc, end='\t\t\t\r')
        #     # if afr > 0:
        #     #     print('send')
        #     #     udp.send(np.array([afr]))

        #     pca_data = X

        #     # Create a 3D scatter plot
        #     # ani = FuncAnimation(fig, update, frames=825, init_func=init, repeat=False)
        #     # plt.show()

        #     # reset count and firing matrix
        #     c = 0
        #     fr_matrix = []
        
        # c += 1
        # fr_matrix.append(fr)
    
except KeyboardInterrupt:
    filename = "matrix_bin.csv"
    header = ["ch_" + str(i) for i in (1,2,3,5,7,13,14,15,17,19,22,23,24,26,28,30,32)]
    # header = ["ch_" + str(i) for i in range(1,33)]
    with open(filename, 'w', newline="") as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow(header)
        for row in fr_export:
            writer.writerow(row)
        



# one channel
# while 1:
#     data = udp.recv()
#     sort_code = 1
#     channel = 26
#     sc = data[sort_code-1][channel-1]

#     print('CHANNEL:', channel, 'SORT:', sort_code, '\t', sc, end='\t\t\t\r')
#     if sc < 7:
#         print('send')
#         udp.send(np.array([sc]))


# udp = tdt.TDTUDP(host=RZ_IP, send_type=np.float32, recv_type=np.float32)

# SEND_PACKETS = 8
# ct = 0
# while 1:
#     ct += 1
#     fakedata = range(ct % 10, SEND_PACKETS + ct % 10)
#     if udp.send_type == float:
#         fakedata = [x * 2. for x in fakedata]

#    data = udp.recv()
#    print(data)


