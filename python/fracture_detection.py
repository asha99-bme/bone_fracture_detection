import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_and_preprocess(img_path: str):
    """Load X-ray and apply basic preprocessing."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {img_path}")
    
    # Histogram equalization to improve contrast
    img_eq = cv2.equalizeHist(img)
    
    # Slight denoising
    img_blur = cv2.GaussianBlur(img_eq, (5, 5), 0)
    
    return img, img_blur


def detect_edges(img_blur):
    """Canny edge detection."""
    edges = cv2.Canny(img_blur, 50, 150)
    return edges


def detect_lines(edges):
    """Probabilistic Hough Transform to detect line segments."""
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=40,
        maxLineGap=10
    )
    if lines is None:
        return []
    return lines


def analyze_lines(lines):
    """Compute angles and lengths for each line segment."""
    angles = []
    lengths = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        angle = np.degrees(np.arctan2(dy, dx))
        length = np.sqrt(dx**2 + dy**2)
        angles.append(angle)
        lengths.append(length)
    return np.array(angles), np.array(lengths)


def find_main_axis_angle(angles, lengths, top_k=5):
    """Estimate main bone axis angle from the longest segments."""
    if len(angles) == 0:
        return 0.0
    idx_sorted = np.argsort(lengths)[::-1]
    top_k = min(top_k, len(idx_sorted))
    main_angles = angles[idx_sorted[:top_k]]
    main_angle = np.median(main_angles)
    return main_angle


def mark_fracture_candidates(orig_img, lines, angles, lengths,
                             main_angle, angle_thresh=15, length_ratio=0.6):
    """Highlight cortical vs suspicious fracture-like segments."""
    if len(lengths) == 0:
        return orig_img

    img_color = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR)
    max_len = np.max(lengths)
    length_thresh = length_ratio * max_len

    for i, line in enumerate(lines):
        x1, y1, x2, y2 = line[0]
        off_angle = abs(angles[i] - main_angle) > angle_thresh
        is_short = lengths[i] < length_thresh

        if off_angle and is_short:
            color = (0, 0, 255)   # red: suspicious
            thickness = 3
        else:
            color = (0, 255, 0)   # green: cortical
            thickness = 1

        cv2.line(img_color, (x1, y1), (x2, y2), color, thickness)

    return img_color


def plot_results(orig, edges, overlay):
    """Show original, edges, and overlay side-by-side."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(orig, cmap='gray')
    axes[0].set_title("Original X-ray")
    axes[0].axis('off')

    axes[1].imshow(edges, cmap='gray')
    axes[1].set_title("Canny Edges")
    axes[1].axis('off')

    axes[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Fracture Candidates (Heuristic)")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()


def main():
    img_path = Path("..") / "images" / "foot_xray_example.jpg"
    orig, img_blur = load_and_preprocess(str(img_path))
    edges = detect_edges(img_blur)
    lines = detect_lines(edges)

    if not lines:
        print("No lines detected. Try adjusting parameters or using another X-ray.")
        return

    angles, lengths = analyze_lines(lines)
    main_angle = find_main_axis_angle(angles, lengths)
    print(f"Estimated main bone axis angle: {main_angle:.2f} degrees")

    overlay = mark_fracture_candidates(orig, lines, angles, lengths, main_angle)
    plot_results(orig, edges, overlay)

    # Save output
    out_path = Path("..") / "results" / "python_output.png"
    out_path.parent.mkdir(exist_ok=True, parents=True)
    cv2.imwrite(str(out_path), overlay)
    print(f"Saved result to {out_path}")


if __name__ == "__main__":
    main()
