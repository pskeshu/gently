"""
Measure centering error after moving to an embryo.

Click on where the embryo actually is - the script will calculate
how far off it is from center (where it should be).
"""

import json

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Load the image after moving to embryo
IMAGE_PATH = "camera_captures/bottom_camera_20251130_175112.jpg"
OUTPUT_FILE = "centering_error.json"

# Expected embryo position (center of 2048x2048 image)
CENTER_X = 1024
CENTER_Y = 1024

# Store clicked position
clicked_pos = [None, None]


def on_click(event):
    if event.xdata is not None and event.ydata is not None:
        clicked_pos[0] = event.xdata
        clicked_pos[1] = event.ydata

        # Calculate error
        error_x = clicked_pos[0] - CENTER_X
        error_y = clicked_pos[1] - CENTER_Y

        # Convert to microns (assuming 10x objective, 6.5um pixel)
        um_per_pixel = 6.5 / 10.0
        error_x_um = error_x * um_per_pixel
        error_y_um = error_y * um_per_pixel

        # Build result
        result = {
            "clicked_x": float(clicked_pos[0]),
            "clicked_y": float(clicked_pos[1]),
            "center_x": CENTER_X,
            "center_y": CENTER_Y,
            "error_x_pixels": float(error_x),
            "error_y_pixels": float(error_y),
            "error_x_direction": "RIGHT" if error_x > 0 else "LEFT",
            "error_y_direction": "BELOW" if error_y > 0 else "ABOVE",
            "error_x_um": float(error_x_um),
            "error_y_um": float(error_y_um),
            "um_per_pixel": um_per_pixel,
        }

        # Write to file
        with open(OUTPUT_FILE, "w") as f:
            json.dump(result, f, indent=2)

        print(f"\n{'=' * 50}")
        print(f"CLICKED POSITION: ({clicked_pos[0]:.1f}, {clicked_pos[1]:.1f})")
        print(f"CENTER POSITION:  ({CENTER_X}, {CENTER_Y})")
        print(f"{'=' * 50}")
        print(f"ERROR X: {error_x:+.1f} pixels ({result['error_x_direction']} of center)")
        print(f"ERROR Y: {error_y:+.1f} pixels ({result['error_y_direction']} center)")
        print(f"{'=' * 50}")
        print(f"ERROR X: {error_x_um:+.1f} um")
        print(f"ERROR Y: {error_y_um:+.1f} um")
        print(f"{'=' * 50}")
        print(f"\nSaved to: {OUTPUT_FILE}")

        # Update the plot with clicked marker
        ax.plot(
            clicked_pos[0],
            clicked_pos[1],
            "go",
            markersize=15,
            markeredgewidth=3,
            markerfacecolor="none",
            label="Actual embryo position",
        )
        ax.legend()
        fig.canvas.draw()


# Load image
img = np.array(Image.open(IMAGE_PATH))

# Create figure
fig, ax = plt.subplots(figsize=(12, 12))
ax.imshow(img, cmap="gray")

# Draw crosshairs at center
ax.axhline(CENTER_Y, color="red", linestyle="--", alpha=0.7, linewidth=1, label="Center")
ax.axvline(CENTER_X, color="red", linestyle="--", alpha=0.7, linewidth=1)
ax.plot(CENTER_X, CENTER_Y, "r+", markersize=30, markeredgewidth=2)

ax.set_title("Click on where EMBRYO 3 actually is\n(Red crosshairs = center where it SHOULD be)")
ax.legend()

# Connect click event
fig.canvas.mpl_connect("button_press_event", on_click)

print("\nClick on where embryo_3 actually appears in the image.")
print("The red crosshairs show the center (where it should be).\n")

plt.show()
