import cv2
import numpy as np

def create_smooth_traversal_no_holes(image_path, output_path, pathway_width=325, remove_obstacles=325):
    """Create smooth traversal paths while filling holes inside lobules"""
    
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    yellow_mask = np.all(img_rgb == [255, 255, 0], axis=2)
    white_mask = np.all(img_rgb == [255, 255, 255], axis=2)
    
    # Step 1: Remove small yellow obstacles (erode yellow regions)
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (remove_obstacles, remove_obstacles))
    yellow_eroded = cv2.erode(yellow_mask.astype(np.uint8), kernel_erode)
    
    # Step 2: Widen white pathways (dilate white regions)
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pathway_width, pathway_width))
    white_dilated = cv2.dilate(white_mask.astype(np.uint8), kernel_dilate)
    
    # Step 3: Fill holes in yellow regions after erosion
    yellow_filled = cv2.morphologyEx(yellow_eroded, cv2.MORPH_CLOSE, kernel_erode)
    
    # Step 4: Create result - prioritize white pathways, fill yellow holes
    result = np.zeros_like(img_rgb)
    result[white_dilated.astype(bool)] = [255, 255, 255]  # White pathways
    result[yellow_filled.astype(bool) & ~white_dilated.astype(bool)] = [255, 255, 0]  # Solid lobules
    
    # Step 5: Fill any remaining gaps with white
    remaining = ~(white_dilated.astype(bool) | yellow_filled.astype(bool))
    result[remaining] = [255, 255, 255]
    
    # Step 6: Fill ALL white holes inside yellow structures with yellow
    # Create a binary image where yellow is 1 and everything else is 0
    yellow_binary = np.zeros((img_rgb.shape[0], img_rgb.shape[1]), dtype=np.uint8)
    yellow_binary[np.all(result == [255, 255, 0], axis=2)] = 1
    
    # Create a binary image where white is 1 and everything else is 0
    white_binary = np.zeros_like(yellow_binary)
    white_binary[np.all(result == [255, 255, 255], axis=2)] = 1
    
    # Create a mask for flood fill (needs to be 2 pixels larger in each dimension)
    h, w = yellow_binary.shape[:2]
    mask = np.zeros((h+2, w+2), dtype=np.uint8)
    
    # Create a copy of white_binary for flood filling
    # We'll use this to identify which white areas are outside (connected to border)
    outside_white = white_binary.copy()
    
    # Flood fill from the border (0,0) to mark all white areas connected to outside
    if outside_white[0, 0] == 0:  # If the corner is not white, start there
        cv2.floodFill(outside_white, mask, (0, 0), 2)
    else:  # If corner is white, find a non-white pixel on border to start
        border_points = [(0, i) for i in range(w)] + [(h-1, i) for i in range(w)] + \
                       [(i, 0) for i in range(h)] + [(i, w-1) for i in range(h)]
        for point in border_points:
            if outside_white[point] == 0:
                cv2.floodFill(outside_white, mask, point, 2)
                break
    
    # Reset the mask for next flood fill
    mask[:] = 0
    
    # Now flood fill from all border points that are white
    for i in range(w):
        if white_binary[0, i] == 1:  # Top border
            cv2.floodFill(outside_white, mask, (i, 0), 2)
        if white_binary[h-1, i] == 1:  # Bottom border
            cv2.floodFill(outside_white, mask, (i, h-1), 2)
    
    for i in range(h):
        if white_binary[i, 0] == 1:  # Left border
            cv2.floodFill(outside_white, mask, (0, i), 2)
        if white_binary[i, w-1] == 1:  # Right border
            cv2.floodFill(outside_white, mask, (w-1, i), 2)
    
    # White holes are white pixels (1) that weren't reached by flood fill (not 2)
    # These are white areas completely surrounded by yellow
    white_holes = (white_binary == 1) & (outside_white == 1)
    
    # Fill these white holes with yellow in the result image
    result[white_holes] = [255, 255, 0]
    
    cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    
    print(f"Created smooth traversal with filled lobules:")
    print(f"- Widened pathways by {pathway_width} pixels")
    print(f"- Removed obstacles up to {remove_obstacles} pixels")
    print(f"- Filled holes in lobules")
    
    return result

# Create smooth traversal without holes
create_smooth_traversal_no_holes(
    "/work/xi47luy/lobule1.jpg",
    "/work/xi47luy/smooth_traversal_no_holes.jpg",
    pathway_width=325,
    remove_obstacles=325
)