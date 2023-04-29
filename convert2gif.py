from PIL import Image
from glob import glob
from natsort import natsorted

# Set the desired frames per second (FPS) for the GIF
fps = 10

# Set the duration of each frame in milliseconds
frame_duration = int(1000 / fps)

image_files = glob('res1/*.png')

# Sort the image file names using natsort
sorted_image_files = natsorted(image_files)

print(sorted_image_files)

# Create a list to hold the frames
frames = []

# Loop through the PNG files and append each frame to the list
for file in sorted_image_files:
    frame = Image.open(file)
    frames.append(frame)

# Save the frames as a GIF file
frames[0].save('output_rgb.gif', format='GIF', append_images=frames[1:], save_all=True, duration=frame_duration, loop=0)