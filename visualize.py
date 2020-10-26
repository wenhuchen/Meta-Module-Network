import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import Constants
import json
import sys
import random
import os

"""
with open('sceneGraphs/val_bounding_box.json') as f:
	data = json.load(f)

for k, v in data.items():
	im = np.array(Image.open("images/{}.jpg".format(k)), dtype=np.uint8)
	# Create figure and axes
	fig, ax = plt.subplots(1)
	ax.imshow(im)
	for bk, bv in v.items():
		# Create a Rectangle patch
		rect = patches.Rectangle((bv['x'],bv['y']),bv['w'],bv['h'],linewidth=1,edgecolor='r',facecolor='none')
		# Add the patch to the Axes
		ax.add_patch(rect)

	plt.show()
	raw_input("Press Enter to continue...")
"""


"""
im = np.array(Image.open("../images/{}.jpg".format(image_id)), dtype=np.uint8)
# Create figure and axes
fig, (ax1, ax2) = plt.subplots(2, 1)

with open('/tmp/proposed.txt') as f:
	ax1.imshow(im)
	ax1.set_title('proposed')
	for line in f:
		x, y, w, h = line.strip().split(' ')
		# Create a Rectangle patch
		rect = patches.Rectangle((int(x), int(y)), int(w), int(h),linewidth=1,edgecolor='r',facecolor='none')
		# Add the patch to the Axes
		ax1.add_patch(rect)

with open('/tmp/groundtruth.txt') as f:
	ax2.imshow(im)
	ax2.set_title('groundtruth')
	for line in f:
		x, y, w, h = line.strip().split(' ')
		# Create a Rectangle patch
		rect = patches.Rectangle((int(x), int(y)), int(w), int(h),linewidth=1,edgecolor='r',facecolor='none')
		# Add the patch to the Axes
		ax2.add_patch(rect)

plt.show()
"""
# image = Image.open('../images/{}.jpg'.format(image_id))
# image.show()
option = sys.argv[1]
if option == "vis":
    print("visualizing image")
    image_id = sys.argv[2]
    position = sys.argv[3]
    position = json.loads(position)
    fig, ax = plt.subplots(1)
    I = np.array(Image.open("../images/{}.jpg".format(image_id)), dtype=np.uint8)
    if position[0] < 1:
        height = I.shape[0]
        width = I.shape[1]
        x = width * position[0]
        y = height * position[1]
        x1 = width * position[2]
        y1 = height * position[3]
    else:
        x, y, x1, y1 = position

    shape = (I.shape[0], I.shape[1], 1)
    A = np.ones(shape) * 0.2
    # for k in range(2):
    A[int(y):int(y1), int(x):int(x1)] = 1

    ax.add_patch(plt.Rectangle((int(x), int(y)),
                               int(x1 - x), int(y1 - y), fill=False,
                               edgecolor='red', linewidth=1))
    A /= np.max(A)
    A = A * I + (1.0 - A) * 255
    A = A.astype('uint8')

    ax.imshow(A, interpolation='bicubic')
    ax.axis('off')
    plt.tight_layout()
    # rect = patches.Rectangle((int(x), int(y)), int(x1 - x), int(y1 - y),linewidth=2, edgecolor='r',facecolor='none')
    # ax.add_patch(rect)
    # ax.axis('off')
    # ax.imshow(im)
    plt.show()
elif option == 'bottom-up':
    print("visualizing image")
    image_id = sys.argv[2]
    if not image_id.startswith('n'):
        if len(image_id) < 7:
            image_id = "0" * (7 - len(image_id)) + image_id
    bottom_up = np.load(os.path.join('../gqa_bottom_up_features/', 'gqa_{}.npz'.format(image_id)))
    adaptive_num_regions = min((bottom_up['conf'] > 0.15).sum(), 48)
    bbox_feat = bottom_up['norm_bb'][:adaptive_num_regions]
    Constants.show_im_bboxes(image_id, bbox_feat)
else:
    scene_graph = sys.argv[2]
    image_id = sys.argv[3]
    if option == "relation":
        with open(scene_graph) as f:
            data = json.load(f)

        for k, v in data[image_id].items():
            if v['relations']:
                for rel in v['relations']:
                    v1 = data[image_id][rel['object']]
                    if rel['name'] not in ['to the left of', 'to the right of']:
                        print(v['name'], (v['x'], v['y'], v['w'], v['h']), rel['name'],
                              v1['name'], (v1['x'], v1['y'], v1['w'], v1['h']),
                              Constants.intersect((v['x'], v['y'], v['w'], v['h']), (v1['x'], v1['y'], v1['w'], v1['h'])))
    if option == "object":
        with open(scene_graph) as f:
            data = json.load(f)
        print("finished loading the data")
        for k, v in data[image_id].items():
            print(k, v['name'], v['attributes'])

    if option == 'bbox':
        with open(scene_graph) as f:
            data = json.load(f)
        print("finished loading the data")
        for k, v in data[image_id].items():
            print(k, v['name'], v['x'], v['y'], v['w'], v['h'])

    if option == 'draw':
        with open(scene_graph) as f:
            data = json.load(f)
        print("finished loading the data")

        fig, ax = plt.subplots(1)
        im = np.array(Image.open("../images/{}.jpg".format(image_id)), dtype=np.uint8)
        ax.imshow(im)

        colors = ['yellow', 'red', 'purple', 'blue', 'green', 'orange']
        for k, v in data[image_id].items():
            if k == '0':
                continue
            # x, y, w, h = line.strip().split(' ')
            # Create a Rectangle patch
            color = random.choice(colors)
            rect = patches.Rectangle((v['x'], v['y']), v['w'], v['h'], linewidth=1, edgecolor=color, facecolor='none')
            # Add the patch to the Axes
            ax.add_patch(rect)
            plt.text(v['x'], v['y'], v['name'], bbox=dict(facecolor=color, alpha=0.5))

        plt.show()
