import cv2
import numpy as np
from reedsolo import RSCodec

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

# cv2.imshow("Processed Square QR Code", square)

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
print(num_rows)

for i in range(num_cols + 1):
    x_line = i * module_size
    cv2.line(color_grid, (x_line, 0), (x_line, height), (0, 255, 0), 1)

for i in range(num_rows + 1):
    y_line = i * module_size
    cv2.line(color_grid, (0, y_line), (width, y_line), (0, 255, 0), 1)

# cv2.imshow("QR Code with Grid", color_grid)

color_contours = cv2.cvtColor(square, cv2.COLOR_GRAY2BGR)
cv2.drawContours(color_contours, contours, -1, (255, 0, 0), 1)
# cv2.imshow("QR Code Contours", color_contours)

_, binary_qr = cv2.threshold(square, 127, 255, cv2.THRESH_BINARY_INV)
# cv2.imshow("binary_qrbinary_qr", binary_qr)

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


# unmasking

ec = qr_matrix[8][0] + 2*qr_matrix[8][1]
mask = qr_matrix[8][2] + 2*qr_matrix[8][3] + 4*qr_matrix[8][4]
unmasked_matrix = []

def apply_mask(mask_id, x, y):
    if mask_id == 0: return (x + y) % 2 == 0 
    if mask_id == 1: return y % 2 == 0 
    if mask_id == 2: return x % 3 == 0
    if mask_id == 3: return (x + y) % 3 == 0
    if mask_id == 4: return (y // 2 + x // 3) % 2 == 0
    if mask_id == 5: return (x * y) % 2 + (x * y) % 3 == 0
    if mask_id == 6: return ((x * y) % 2 + (x * y) % 3) % 2 == 0
    if mask_id == 7: return ((x + y) % 2 + (x * y) % 3) % 2 == 0


def get_alignment_positions(version):
    """
    Returns the alignment pattern positions for a given QR code version.
    """
    if version < 2:
        return [] 

    # Alignment pattern positions for versions 2 up to 40
    alignment_table = [
        [],  
        [6, 22],  
        [6, 26],  
        [6, 30],  
        [6, 34],  
        [6, 22, 38],  
        [6, 24, 42],  
        [6, 26, 46],  
        [6, 28, 50],  
        [6, 30, 54],  
        [6, 32, 58],  
        [6, 34, 62],  
        [6, 26, 46, 66],  
        [6, 26, 48, 70],  
        [6, 26, 50, 74],  
        [6, 30, 54, 78],  
        [6, 30, 56, 82],  
        [6, 30, 58, 86],  
        [6, 34, 62, 90],  
        [6, 28, 50, 72, 94],  
        [6, 26, 50, 74, 98],  
        [6, 30, 54, 78, 102],  
        [6, 28, 54, 80, 106],  
        [6, 32, 58, 84, 110],  
        [6, 30, 58, 86, 114],  
        [6, 34, 62, 90, 118],  
        [6, 26, 50, 74, 98, 122],  
        [6, 30, 54, 78, 102, 126],  
        [6, 26, 52, 78, 104, 130],  
        [6, 30, 56, 82, 108, 134],  
        [6, 34, 60, 86, 112, 138],  
        [6, 30, 58, 86, 114, 142],  
        [6, 34, 62, 90, 118, 146],  
        [6, 30, 54, 78, 102, 126, 150],  
        [6, 24, 50, 76, 102, 128, 154],  
        [6, 28, 54, 80, 106, 132, 158],  
        [6, 32, 58, 84, 110, 136, 162],  
        [6, 26, 54, 82, 110, 138, 166],  
        [6, 30, 58, 86, 114, 142, 170],  
        [6, 34, 62, 90, 118, 146, 174],  
    ]

    return alignment_table[version - 1]

def is_data_region(i, j, version):

    # Position markers
    if (i < 9 and j < 9) or (i < 9 and j > (version * 4 + 9 - 10)) or (i > (version * 4 + 9 - 10) and j < 9):
        return False
    
    # Timing patterns 
    if i == 6 or j == 6:
        return False

    # Format information
    if (i < 9 and (j > 8 and j < (version * 4 + 9 - 8))) or (j < 9 and (i > 8 and i < (version * 4 + 9 - 8))):
        return False

    # Alignment patterns
    
    alignment_positions = get_alignment_positions(version)
    for pos in alignment_positions:
        if (i >= pos - 2 and i <= pos + 2) and (j >= pos - 2 and j <= pos + 2):
            return False
        
    return True

def extract_data_bits(qr_matrix, version, mask_pattern):
    size = len(qr_matrix)
    bits = []
    direction = -1  # Start moving upward
    col = size - 1  # Start from right-most column

    while col > 0:
        if col == 6:  # Skip timing pattern column
            col -= 1
            continue

        for i in range(size):
            row = (size - 1 - i) if direction == -1 else i
            
            # Right column (current)
            if is_data_region(row, col, version):
                bits.append(qr_matrix[row][col] ^ (1 if apply_mask(mask_pattern, row, col) else 0))
            
            # Left column (current - 1)
            if col > 0 and is_data_region(row, col-1, version):
                bits.append(qr_matrix[row][col-1] ^ (1 if apply_mask(mask_pattern, row, col-1) else 0))

        direction *= -1  # Reverse direction
        col -= 2

    return bits
    
version = ((num_rows - 21) // 4) + 1

unmasked_matrix = extract_data_bits(qr_matrix, version, mask)

def bit_matrix_to_symbols(bit_matrix):
    
    # flat_bits = bit_matrix.flatten()
    flat_bits = bit_matrix
    # Group the bits into bytes (8 bits each)
    bytes_list = []
    for i in range(0, len(flat_bits), 8):
        # Extract 8 bits (or fewer if at the end of the array)
        byte_bits = flat_bits[i:i + 8]
        
        # If there are fewer than 8 bits, pad with zeros
        if len(byte_bits) < 8:
            byte_bits = np.pad(byte_bits, (0, 8 - len(byte_bits)), mode='constant')
        
        # Convert the 8 bits into a single byte
        byte = 0
        for j in range(8):
            byte |= (byte_bits[j] << (7 - j))  # Shift and combine bits into a byte
        
        bytes_list.append(byte.astype(np.uint8))
    
    return bytes_list

# print(unmasked_matrix)
# unmasked_matrix = [item for line in unmasked_matrix for item in line]
unmasked_matrix = np.array(unmasked_matrix)
print(unmasked_matrix, len(unmasked_matrix))
symbols = bit_matrix_to_symbols(unmasked_matrix)
print(symbols)


def calculate_nsym(version, ec_level):
    # Total number of codewords for the given version
    total_codewords = 4 * version + 17 * (version - 1) + 1

    if ec_level == 0:
        data_codewords = total_codewords - 7
    elif ec_level == 1:
        data_codewords = total_codewords - 10
    elif ec_level == 2:
        data_codewords = total_codewords - 13
    elif ec_level == 3:
        data_codewords = total_codewords - 17

    # Calculate nsym
    nsym = total_codewords - data_codewords
    return nsym

nysm = calculate_nsym(version, ec)
rsc = RSCodec(nysm)
print(nysm, ec, version, num_rows)
decoded_data = rsc.decode(symbols)
decoded_bytes = bytes(symbols)
decoded_text = decoded_bytes.decode('utf-8')
print(decoded_text)

