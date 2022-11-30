import numpy as np
import os
import imageio
import matplotlib.pyplot as plt
from IPython.display import Image, HTML, display
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import rcParams


def resize_show_image(image):
    min_d1_array = (image != 0).argmax(axis=0)
    max_d1_array = (np.flip(image, axis=0) != 0).argmax(axis=0)
    min_d2_array = (image != 0).argmax(axis=1)
    max_d2_array = (np.flip(image, axis=1) != 0).argmax(axis=1)
    min_d1 = np.min(min_d1_array[min_d1_array != 0])
    max_d1 = image.shape[0] - np.min(max_d1_array[max_d1_array != 0]) + 1
    min_d2 = np.min(min_d2_array[min_d2_array != 0])
    max_d2 = image.shape[1] - np.min(max_d2_array[max_d2_array != 0]) + 1
    return min_d1, max_d1, min_d2, max_d2


def save_and_show_flow_gif(displacement, mask, path, filename, frame_duration=0.15, is_show=False):
    frame_num = displacement.shape[-1]
    min_d1, max_d1, min_d2, max_d2 = resize_show_image(mask)
    mask = np.array(np.stack([mask, mask], axis=2), dtype=bool)
    full_filename = os.path.join(path, filename)

    with imageio.get_writer(full_filename, mode='I', duration=frame_duration) as writer:
        for frame in range(frame_num):
            flow = displacement[0, :, :, :, frame].squeeze().transpose(1, 2, 0) * mask
            flow = flow[min_d1-4:max_d1+5, min_d2-4:max_d2+5, :]
            flow = flow[::-1, :, :]
            # fig, _ = ne.plot.flow([flow], width=5, show=show)
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.set_facecolor('black')
            u, v = flow[..., 0], flow[..., 1]
            ax.quiver(u, -v, color='y', angles='xy', units='xy', scale=1, width=0.15, linewidths=1, minlength=0)
            if is_show:
                plt.show()
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            writer.append_data(image)
            plt.close('all')
    writer.close()

    with open(full_filename, 'rb') as f:
        display(Image(data=f.read(), width=400, height=400, format='png'))


def save_and_show_gif(image_to_show, path, filename, frame_duration=0.15):
    full_filename = os.path.join(path, filename)

    frame_num = image_to_show.shape[-1]
    max_value = np.max(image_to_show)
    with imageio.get_writer(full_filename, mode='I', duration=frame_duration) as writer:
        for frame in range(frame_num):
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            plt.imshow(image_to_show[:, :, frame], cmap='gray', vmin=0, vmax=max_value)
            plt.subplots_adjust(hspace=0, wspace=0)
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            writer.append_data(image)
            plt.close('all')
    writer.close()

    with open(full_filename, 'rb') as f:
        display(Image(data=f.read(), width=300, height=300, format='png'))

