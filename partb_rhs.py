import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


class RhsToothExtractor:
    """
    Extracts the rectangular tooth/plate region from the RIGHT-HAND-SIDE (RHS) camera image.
    
    Main steps:
    1. Perspective correction (warp) to straighten the view
    2. Fixed column crop
    3. Bilateral filter → threshold → Sobel edges → erode
    4. Detect vertical edge **rises** (gaps between plates - RHS direction)
    5. Select valid plate region (prefer two points, fallback to single + offset)
    """
    
    def __init__(self):
        # -- initial Background remove make the image as 1200 x 1200 --
        self.original_image_row_start = 400
        self.original_image_row_end   = 1600

        # ── Perspective correction parameters ──
        # These values usually need to be tuned differently from LHS
        self.col_min_top    = 650          # ← adjust these for RHS camera
        self.col_max_top    = 850
        self.col_min_bot    = 480
        self.col_max_bot    = 680
        self.row_min        = 0
        self.row_max        = 1200
        
        self.column_crop_min = 100
        self.column_crop_max = 1100
        
        # After warp → final width
        self.warped_width = self.col_max_bot - self.col_min_bot   # ≈ 260–280

        # ── Edge detection & plate selection parameters ──
        self.thresh_value     = 100
        self.bilateral_d      = 5
        self.bilateral_sigmaColor = 25
        self.bilateral_sigmaSpace = 3
        
        self.sobel_ksize      = 5
        self.erode_kernel     = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        
        self.rise_threshold   = 5.0          # looking for upward rises (rhs)
        self.min_plate_dist   = 60
        self.max_plate_dist   = 120
        self.default_spacing  = 150           # used in single-point fallback
        
        # We usually crop from the lowest detected point downward
        self.final_crop_top_margin = 0       # can increase if needed

    def _warp_image(self, img: np.ndarray) -> np.ndarray:
        """Apply perspective transform to straighten the view"""
        if img is None or img.size == 0:
            return np.zeros((self.row_max, self.warped_width, 3), dtype=np.uint8)

        # Source points - usually mirrored/shifted compared to LHS
        pts_src = np.float32([
            [self.col_min_top, self.row_min],     # top-left
            [self.col_max_top, self.row_min],     # top-right
            [self.col_min_bot, self.row_max],     # bottom-left
            [self.col_max_bot, self.row_max],     # bottom-right
        ])

        pts_dst = np.float32([
            [0, 0],
            [self.warped_width, 0],
            [0, self.row_max],
            [self.warped_width, self.row_max],
        ])

        matrix = cv.getPerspectiveTransform(pts_src, pts_dst)
        warped = cv.warpPerspective(img, matrix, (self.warped_width, self.row_max))
        
        # Fixed column crop after warp
        warped = warped[ self.column_crop_min:self.column_crop_max,:]
        

        
        return warped

    def _find_plate_edge_points(self, warped: np.ndarray) -> list[int]:
        """Detect the main **rises** (gaps) in the top edge of the plates - RHS"""
        if warped is None or warped.size == 0:
            return []

        # Tight crop — we mostly care about the top part
        tight = warped[:100, :, :]

        bilateral = cv.bilateralFilter(
            tight,
            d=self.bilateral_d,
            sigmaColor=self.bilateral_sigmaColor,
            sigmaSpace=self.bilateral_sigmaSpace
        )

        gray = bilateral[:, :, 1]  # green channel often good contrast
        _, thresh = cv.threshold(gray, self.thresh_value, 250, cv.THRESH_BINARY)

        # Sobel magnitude
        sobelx = cv.Sobel(thresh, cv.CV_64F, 1, 0, ksize=self.sobel_ksize)
        sobely = cv.Sobel(thresh, cv.CV_64F, 0, 1, ksize=self.sobel_ksize)
        sobel_mag = cv.magnitude(sobelx, sobely)

        # Clean a bit
        edge_clean = cv.erode(sobel_mag, self.erode_kernel)
        
  

        # Find the FIRST (top-most) strong edge pixel in each column
        edge_y = np.argmax(edge_clean > 0, axis=0).astype(float)

        # Mask columns with no edge
        has_edge = np.any(edge_clean > 0, axis=0)
        edge_y[~has_edge] = np.nan

        # Look for RISES (positive diff → going up) — this is the key difference from LHS
        dy = np.diff(edge_y)
        rise_locations = np.where(dy > self.rise_threshold)[0]
        
        return rise_locations.tolist()

    def _select_plate_region(self, rise_points: list[int], img_width: int) -> tuple[int, int]:
        """
        From detected rise points → decide left & right x coordinates of the plate.
        Prefer two real rises if distance is reasonable, else fallback to single + offset.
        """
        if len(rise_points) >= 2:
            p1, p2 = sorted(rise_points[:2])  # take first two
            dist = p2 - p1
            if self.min_plate_dist <= dist <= self.max_plate_dist:
                return p1, p2

        # Fallback: single point or invalid pair
        if len(rise_points) >= 1:
            single_x = rise_points[0]
        else:
            single_x = img_width // 2  # worst-case fallback

        dist_left = single_x
        dist_right = img_width - single_x

        # For RHS — we usually prefer to extend toward the **left** when uncertain
        if dist_left > dist_right:
            second_x = single_x - self.default_spacing   # extend left
        else:
            second_x = single_x + self.default_spacing   # extend right

        second_x = np.clip(second_x, 0, img_width - 1)

        x_left  = min(single_x, second_x)
        x_right = max(single_x, second_x)

        return x_left, x_right

    def extract(self, original_image: np.ndarray) -> np.ndarray:
        if original_image is None or original_image.size == 0:
            return np.array([])
        
        original_image = original_image[:, self.original_image_row_start:self.original_image_row_end, :]
        
        # Step 1: Perspective correction + column crop
        warped = self._warp_image(original_image)

        if warped.size == 0:
            return np.array([])

        # Step 2: Detect gap locations (drops or rises)
        # (For LHS: drop_points; For RHS: rise_points)
        points = self._find_plate_edge_points(warped)  # Renamed for generality

        # Step 3: Decide left-right boundaries of the plate
        x_left, x_right = self._select_plate_region(points, warped.shape[1])

        # Step 4: Recompute edge_y (top edge positions) on the tight top region
        # (We need the actual y-values now, within x_left:x_right)
        tight = warped[:100, :, :]
        bilateral = cv.bilateralFilter(
            tight,
            d=self.bilateral_d,
            sigmaColor=self.bilateral_sigmaColor,
            sigmaSpace=self.bilateral_sigmaSpace
        )
        gray = bilateral[:, :, 1]
        _, thresh = cv.threshold(gray, self.thresh_value, 250, cv.THRESH_BINARY)
        sobelx = cv.Sobel(thresh, cv.CV_64F, 1, 0, ksize=self.sobel_ksize)
        sobely = cv.Sobel(thresh, cv.CV_64F, 0, 1, ksize=self.sobel_ksize)
        sobel_mag = cv.magnitude(sobelx, sobely)
        edge_clean = cv.erode(sobel_mag, self.erode_kernel)
        
        edge_y = np.argmax(edge_clean > 0, axis=0).astype(float)
        has_edge = np.any(edge_clean > 0, axis=0)
        edge_y[~has_edge] = np.nan
        
        # Now set top_y to the max (lowest) edge_y in the plate region
        plate_edge_y = edge_y[x_left:x_right]
        if np.any(~np.isnan(plate_edge_y)):
            top_y = int(np.nanmax(plate_edge_y)) + self.final_crop_top_margin
        else:
            top_y = 0  # Fallback if no edges detected

        cropped = warped[top_y:, x_left:x_right]

        if cropped.size == 0:
            return warped

        return cropped


# ────────────────────────────────────────
#   Example usage
# ────────────────────────────────────────
if __name__ == "__main__":
    extractor = RhsToothExtractor()

    # Replace with your RHS camera image path
    img = cv.imread("/home/dell/ZF/ZF/31.1.2026/6partb/20260131_162025/25320882/14.png")
    
    if img is not None:
        result = extractor.extract(img)
        
        plt.figure(figsize=(6,10))
        plt.imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))
        plt.title("RHS Extracted Tooth")
        plt.axis('off')
        plt.show()
    else:
        print("Could not load image.")