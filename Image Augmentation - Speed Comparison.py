import image_edit
import shutil
import glob
import os

fdirs = ['F:/### Sanabil Dissertation/Image Dataset/Random/input',
         'F:/### Sanabil Dissertation/Image Dataset/Random/output',
         'F:/### Sanabil Dissertation/Image Dataset/Random/split']

import time

start = time.process_time()

# image_edit.Circler(fdirs[0], fdirs[1], transparent=False)
# image_edit.Grabcutter(fdirs[0], fdirs[1], iterations=1, transparent=False)
# image_edit.Any2PNG(fdirs[0], fdirs[1])
# image_edit.Cropper(fdirs[0], fdirs[1])
# image_edit.Flipper(fdirs[0], fdirs[1], direction = 'both')
# image_edit.Rotator(fdirs[0], fdirs[1], step_size=36)
# image_edit.Noisy(fdirs[0], fdirs[1])
# image_edit.Padder(fdirs[0], fdirs[1], padding = 700)
# image_edit.Transparenter([fdirs[2]], fdirs[1])
# image_edit.Splitter(fdirs[0], fdirs[1], 0.8)
# image_edit.Rescaler([fdirs[0]], fdirs[1], scale_factor=0.5)

print(time.process_time() - start)