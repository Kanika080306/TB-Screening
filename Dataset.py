from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

IMAGE_DIR = Path(r"C:\Users\Kanika\Desktop\TB\Montgomery\Montgomery\img")
MASK_DIR = Path(r"C:\Users\Kanika\Desktop\TB\Montgomery\Montgomery\mask")

image_path = next(IMAGE_DIR.glob("*.png"))
mask_path = next(MASK_DIR.rglob(image_path.name))

image = np.array(Image.open(image_path).convert("L"))
mask = np.array(Image.open(mask_path).convert("L"))
mask = (mask > 0).astype(np.uint8)

original = image.copy()
black_bg = image * mask
gray_bg = np.where(mask == 1, image, 128).astype(np.uint8)

fig, ax = plt.subplots(1, 3, figsize=(12, 4))

for a, img, title in zip(
    ax,
    [original, black_bg, gray_bg],
    ["Original", "Black background", "Gray background"]
):
    a.imshow(img, cmap="gray", vmin=0, vmax=255)
    a.set_title(title)
    a.axis("off")

plt.tight_layout()
plt.show()

# Normalize pixel intensities
normalized = black_bg.astype(np.float32) / 255.0

fig, ax = plt.subplots(1, 2, figsize=(9, 4))

ax[0].imshow(black_bg, cmap="gray", vmin=0, vmax=255)
ax[0].set_title("Original intensity (0–255)")

ax[1].imshow(normalized, cmap="gray", vmin=0, vmax=1)
ax[1].set_title("Normalized intensity (0–1)")

for a in ax:
    a.axis("off")

plt.tight_layout()
plt.show()

print("Original range:", black_bg.min(), "to", black_bg.max())
print("Normalized range:", normalized.min(), "to", normalized.max())

SIZE = 224

# Direct resizing
direct = Image.fromarray(black_bg).resize((SIZE, SIZE))

# Preserve aspect ratio and pad
img = Image.fromarray(black_bg)
img.thumbnail((SIZE, SIZE))

preserved = Image.new("L", (SIZE, SIZE), 0)
x = (SIZE - img.width) // 2
y = (SIZE - img.height) // 2
preserved.paste(img, (x, y))

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(direct, cmap="gray")
plt.title("Direct resize")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(preserved, cmap="gray")
plt.title("Aspect ratio preserved")
plt.axis("off")

plt.tight_layout()
plt.show()

# 1. Original X-ray
original = image.copy()

# 2. Segment lungs
segmented = image * mask

# 3. Normalize intensity
normalized = segmented.astype(np.float32) / 255.0

# 4. Resize while preserving aspect ratio
resized = Image.fromarray(
    (normalized * 255).astype(np.uint8)
)
resized.thumbnail((224, 224))

final_img = Image.new("L", (224, 224), 0)
x = (224 - resized.width) // 2
y = (224 - resized.height) // 2
final_img.paste(resized, (x, y))

# Display comparison
fig, ax = plt.subplots(1, 3, figsize=(12, 4))

ax[0].imshow(original, cmap="gray")
ax[0].set_title("Original X-ray")

ax[1].imshow(segmented, cmap="gray")
ax[1].set_title("Lung segmentation")

ax[2].imshow(final_img, cmap="gray")
ax[2].set_title("Combined preprocessing")

for a in ax:
    a.axis("off")

plt.tight_layout()
plt.show()

print("Final image dimensions:", final_img.size)
print("Final intensity range:", np.array(final_img).min(),
      "to", np.array(final_img).max())

image_paths = sorted(IMAGE_DIR.glob("*.png"))
errors = []

for p in image_paths:
    try:
        img = Image.open(p).convert("L")
        mask_path = next(
            (m for m in MASK_DIR.rglob("*.png") if m.stem == p.stem),
            None
        )

        if mask_path is None:
            errors.append(f"{p.name}: missing mask")
            continue

        mask_img = Image.open(mask_path).convert("L")

        if img.size != mask_img.size:
            errors.append(f"{p.name}: dimension mismatch")
        elif np.array(mask_img).max() == 0:
            errors.append(f"{p.name}: empty mask")

    except Exception as e:
        errors.append(f"{p.name}: {e}")

print("Total X-rays:", len(image_paths))
print("Validation errors:", len(errors))

for error in errors:
    print(error)
