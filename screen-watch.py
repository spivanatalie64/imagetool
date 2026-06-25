#!/usr/bin/env python3
"""
screen-watch — Screen capture + analysis tool.
Captures screen regions and processes them with OpenCV.
"""
import cv2
import numpy as np
import sys
import os
import time
import subprocess
from pathlib import Path

HELP = """Usage: screen-watch [command] [args...]

Commands:
  capture [out.png]            Take screenshot of entire screen
  capture-region x y w h [o]   Capture a specific region
  watch [interval]             Watch screen, show changes
  bg-remove [in] [out]         Remove background from image
  describe [in]                Describe image content in text
  help                         This help
"""

def screenshot(path="/tmp/screen.png"):
    """Take a screenshot using available tool."""
    for tool, args in [
        ("gnome-screenshot", ["-f", path]),
        ("scrot", [path]),
        ("import", [path]),  # ImageMagick
        ("spectacle", ["-b", "-n", "-o", path]),
        ("flameshot", ["full", "-p", path]),
    ]:
        try:
            subprocess.run([tool] + args, check=True, timeout=10,
                         capture_output=True)
            if os.path.exists(path) and os.path.getsize(path) > 1000:
                return cv2.imread(path)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None

def describe_image(img):
    """Describe image in text for AI processing."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Divide into 4x4 grid, describe each cell
    cells = []
    for row in range(4):
        for col in range(4):
            y1, y2 = row*h//4, (row+1)*h//4
            x1, x2 = col*w//4, (col+1)*w//4
            cell = img[y1:y2, x1:x2]
            mean = cv2.mean(cell)[:3]
            std = cv2.meanStdDev(cell)[1][:3].flatten()
            edges = cv2.Canny(cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY), 50, 150)
            edge_density = np.sum(edges) / 255 / (cell.shape[0] * cell.shape[1])
            
            cells.append({
                "row": row, "col": col,
                "bgr": (int(mean[0]), int(mean[1]), int(mean[2])),
                "std": (int(std[0]), int(std[1]), int(std[2])),
                "edges": edge_density,
            })
    
    # Find dominant colors
    pixels = img.reshape(-1, 3)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(pixels.astype(np.float32), 5, None,
                                     criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    colors = [{"bgr": (int(c[0]), int(c[1]), int(c[2])),
               "pct": float(np.sum(labels == i) / len(labels))}
              for i, c in enumerate(centers)]
    colors.sort(key=lambda x: -x["pct"])
    
    # Build description
    desc = f"IMAGE: {w}x{h}px\n"
    
    desc += "\nGRID (4x4, BGR colors, edge density):\n"
    for row in range(4):
        for col in range(4):
            c = cells[row*4+col]
            desc += f"  [{row},{col}] BGR=({c['bgr'][0]},{c['bgr'][1]},{c['bgr'][2]})"
            desc += f" edges={c['edges']:.2f}"
        desc += "\n"
    
    desc += "\nDOMINANT COLORS:\n"
    for c in colors[:5]:
        desc += f"  {c['pct']*100:.0f}% BGR=({c['bgr'][0]},{c['bgr'][1]},{c['bgr'][2]})\n"
    
    if w > 10 and h > 10:
        # Edge detection for subject boundaries
        edges = cv2.Canny(gray, 30, 100)
        # Find largest connected region (likely subject)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, bw, bh = cv2.boundingRect(largest)
            desc += f"\nSUBJECT (largest contour): ({x},{y})-{bw}x{bh}\n"
    
    return desc

def cmd_capture(args):
    path = args[0] if args else "/tmp/screen.png"
    img = screenshot(path)
    if img is not None:
        print(f"Captured: {path}")
        print(describe_image(img))
    else:
        print("No screenshot tool available. Install gnome-screenshot, scrot, or import.")

def cmd_capture_region(args):
    x, y, w, h = int(args[0]), int(args[1]), int(args[2]), int(args[3])
    path = args[4] if len(args) > 4 else "/tmp/region.png"
    img = screenshot(path)
    if img is not None:
        region = img[y:y+h, x:x+w]
        cv2.imwrite(path, region)
        print(f"Region captured: {path}")
        print(describe_image(region))
    else:
        print("Failed to capture")

def cmd_describe(args):
    path = args[0]
    img = cv2.imread(path)
    if img is None:
        print(f"Cannot read: {path}")
        return
    print(describe_image(img))

def cmd_bg_remove(args):
    inp = args[0]
    out = args[1] if len(args) > 1 else os.path.splitext(inp)[0] + "_nobg.png"
    img = cv2.imread(inp)
    h, w = img.shape[:2]
    
    mask = np.zeros((h,w), np.uint8)
    bgd = np.zeros((1,65), np.float64)
    fgd = np.zeros((1,65), np.float64)
    margin = int(min(w,h) * 0.15)
    rect = (margin, margin, w-2*margin, h-2*margin)
    
    cv2.grabCut(img, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    fg = np.where((mask==cv2.GC_FGD)|(mask==cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    
    kernel = np.ones((5,5), np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
    fg = cv2.GaussianBlur(fg.astype(np.float32), (15,15), 7)
    
    mf = fg[:,:,None] / 255.0
    result = (img.astype(np.float32) * mf).astype(np.uint8)
    cv2.imwrite(out, result, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    print(f"Saved: {out}")

def cmd_watch(args):
    interval = float(args[0]) if args else 2.0
    print(f"Watching every {interval}s. Ctrl+C to stop.")
    print("---")
    try:
        while True:
            img = screenshot("/tmp/screen_watch.png")
            if img is not None:
                os.system('clear')
                print(describe_image(img))
                print(f"\n[{time.strftime('%H:%M:%S')}] ---")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped.")

def main():
    if len(sys.argv) < 2:
        print(HELP); sys.exit(1)
    cmd = sys.argv[1]
    args = sys.argv[2:]
    cmds = {"capture": cmd_capture, "capture-region": cmd_capture_region,
            "describe": cmd_describe, "bg-remove": cmd_bg_remove,
            "watch": cmd_watch, "help": lambda a: print(HELP)}
    if cmd not in cmds:
        print(f"Unknown: {cmd}"); print(HELP); sys.exit(1)
    cmds[cmd](args)

if __name__ == "__main__":
    main()
