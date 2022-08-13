import numpy as np


cnn_pixel = np.load('basicsr/metrics/cnn_pixel.npy',allow_pickle=True)

for current_iter in cnn_pixel.item():

    loss = cnn_pixel.item()[str(current_iter)]
    current_iter = int(current_iter)
    avg_next = 0.0
    avg_previous = 0.0
    sum_loss = 0.0

    if current_iter % 200 != 0.0:

        sum_loss = sum_loss + loss

    else:
        print(sum_loss)
        if avg_previous == 0.0:
            avg_previous = sum_loss / 2.0

        else:
            avg_next = sum_loss / 2.0

        if (avg_previous != 0.0) and (avg_next != 0.0):
            print(avg_previous, avg_next)
            rate = (avg_previous - avg_next) / avg_previous
            avg_previous = 0.0
            avg_next = 0.0
            sum_loss = 0.0

            print("current iteration:",current_iter,"pixel-wise loss:", rate)

# print(cnn_pixel)