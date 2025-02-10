import cv2
import numpy as np

img = cv2.imread('frame.png', cv2.IMREAD_GRAYSCALE)
if img is None:
    raise ValueError("Image not found or unable to load.")

_, thresh = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)

# Invert the image so that non-white regions are nonzero.
non_white = cv2.findNonZero(255 - thresh)
if non_white is None:
    raise ValueError("No non-white pixels found; check the image or threshold parameters.")

# Get bounding rectangle around the non-white (QR code) area.
x, y, w, h = cv2.boundingRect(non_white)
cropped = img[y:y+h, x:x+w]

(h_c, w_c) = cropped.shape
if h_c != w_c:
    size = max(h_c, w_c)
    # Create a new white square image.
    square = np.ones((size, size), dtype=np.uint8) * 255
    y_offset = (size - h_c) // 2
    x_offset = (size - w_c) // 2
    square[y_offset:y_offset+h_c, x_offset:x_offset+w_c] = cropped
else:
    square = cropped

cv2.imshow("Processed Square QR Code", square)

_, qr_thresh = cv2.threshold(square, 127, 255, cv2.THRESH_BINARY_INV)

# Find contours in the thresholded image.
contours, _ = cv2.findContours(qr_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

module_sizes = []
for cnt in contours:
    cx, cy, cw, ch = cv2.boundingRect(cnt)
    if cw > 3 and ch > 3 and abs(cw - ch) < 3:
        module_sizes.append(cw)

if not module_sizes:
    raise ValueError("No module candidates found. Check the thresholding parameters or image quality.")

module_size = min(module_sizes)
print("Detected module size:", module_size)

# Create a color image for drawing the grid.
color_grid = cv2.cvtColor(square, cv2.COLOR_GRAY2BGR)
height, width = square.shape

# Calculate the number of modules (grid cells).
num_cols = width // module_size
num_rows = height // module_size


for i in range(num_cols + 1):
    x_line = i * module_size
    cv2.line(color_grid, (x_line, 0), (x_line, height), (0, 255, 0), 1)

for i in range(num_rows + 1):
    y_line = i * module_size
    cv2.line(color_grid, (0, y_line), (width, y_line), (0, 255, 0), 1)

cv2.imshow("QR Code with Grid", color_grid)

color_contours = cv2.cvtColor(square, cv2.COLOR_GRAY2BGR)
cv2.drawContours(color_contours, contours, -1, (255, 0, 0), 1)
cv2.imshow("QR Code Contours", color_contours)

_, binary_qr = cv2.threshold(square, 127, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("binary_qrbinary_qr", binary_qr)

# make normalized matrix
qr_matrix = []
for i in range(num_rows):
    row_vals = []
    for j in range(num_cols):

        cell = binary_qr[i*module_size:(i+1)*module_size, j*module_size:(j+1)*module_size]

        avg_val = np.mean(cell)

        if avg_val > 127:
            row_vals.append(1)
        else:
            row_vals.append(0)
    qr_matrix.append(row_vals)

print("QR Code Matrix:")
for row in qr_matrix:
    print(row)

# verify matrix

scale_factor = 10 
qr_matrix_np = np.array(qr_matrix, dtype=np.uint8)

debug_image = 255 * (1 - qr_matrix_np)

debug_image = cv2.resize(debug_image, (num_cols*scale_factor, num_rows*scale_factor), interpolation=cv2.INTER_NEAREST)

cv2.imshow("QR Code from Matrix", debug_image)

cv2.waitKey(0)
cv2.destroyAllWindows()



