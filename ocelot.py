# OCELOT - OCR Correction Tool
# Combined Code with All Features
# Features:
# 1) EasyOCR for text recognition
# 2) Webcam preview and capture
# 3) Undo/Redo
# 4) Scrolling (horizontal + vertical), SHIFT+Scroll for horizontal
# 5) Zoom In/Out (buttons), plus CTRL+Scroll for zoom
# 6) Click-to-edit bounding box text
# 7) CSV/JSON export
# 8) Simple Theming with ttk

# Required dependencies:
#   pip install opencv-python pillow easyocr

import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog, scrolledtext, simpledialog, messagebox
import cv2
from PIL import Image, ImageTk
import easyocr
import csv
import json

# Initialize EasyOCR Reader once to improve efficiency
reader = easyocr.Reader(["en"])

# Global variables
img = None          # Original image as NumPy array (BGR -> RGB)
zoom_img = None     # Resized image for Zoom In/Out
text_data = []      # OCR result storage
undo_stack = []
redo_stack = []
scale_factor = 1.0  # For Zoom In/Out

########################################################
# LOAD/WEBCAM
########################################################

def load_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")]
    )
    if not file_path:
        return

    global img, scale_factor
    scale_factor = 1.0
    img_bgr = cv2.imread(file_path)
    if img_bgr is None:
        messagebox.showerror("Error", "Failed to load image.")
        return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = img_rgb
    process_image(img)

def webcam_preview():
    cap = cv2.VideoCapture(0)
    preview_window = tk.Toplevel(tk_root)
    preview_window.title("Webcam Preview")

    lbl_preview = ttk.Label(preview_window)
    lbl_preview.pack()

    def show_frame():
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_tk = ImageTk.PhotoImage(frame_pil)
            lbl_preview.imgtk = frame_tk
            lbl_preview.configure(image=frame_tk)
            lbl_preview.after(10, show_frame)
        else:
            cap.release()

    def capture():
        ret, frame = cap.read()
        if ret:
            global img, scale_factor
            scale_factor = 1.0
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = img_rgb
            process_image(img)
        cap.release()
        preview_window.destroy()

    btn_capture = ttk.Button(preview_window, text="📸 Capture Image", command=capture)
    btn_capture.pack()

    show_frame()

########################################################
# OCR + PROCESS
########################################################

def process_image(image):
    global text_data
    undo_stack.clear()
    redo_stack.clear()

    results = reader.readtext(image, detail=True)

    text_data = []
    for bbox, text, confidence in results:
        xy_a = tuple(map(int, bbox[0]))  # top-left
        xy_b = tuple(map(int, bbox[2]))  # bottom-right
        text_data.append({
            "index": len(text_data),
            "text": text,
            "confidence": confidence,
            "xy_a": xy_a,
            "xy_b": xy_b
        })

    draw_boxes()
    display_text()

########################################################
# DRAW + ZOOM + SCROLL
########################################################
from functools import partial

canvas = None
canvas_scroll_x = None
canvas_scroll_y = None

import math

def draw_boxes():
    global zoom_img

    if img is None:
        return
    # Apply zoom
    w = int(img.shape[1] * scale_factor)
    h = int(img.shape[0] * scale_factor)

    zoom_img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    # Draw bounding boxes
    display_img = zoom_img.copy()
    for item in text_data:
        idx = item["index"]
        x_min, y_min = item["xy_a"]
        x_max, y_max = item["xy_b"]
        confidence = item["confidence"]

        # Scale bounding box coordinates
        x_min_s = int(x_min * scale_factor)
        y_min_s = int(y_min * scale_factor)
        x_max_s = int(x_max * scale_factor)
        y_max_s = int(y_max * scale_factor)

        # Color by confidence
        if confidence > 0.85:
            color = (0, 255, 0)
        elif confidence < 0.5:
            color = (0, 0, 255)
        else:
            color = (255, 255, 0)

        cv2.rectangle(display_img, (x_min_s, y_min_s), (x_max_s, y_max_s), color, 2)
        cv2.putText(display_img, f'{idx}', (x_min_s, y_min_s - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    pil_img = Image.fromarray(display_img)
    tk_img = ImageTk.PhotoImage(pil_img)

    canvas.delete("all")
    canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)
    canvas.config(scrollregion=canvas.bbox(tk.ALL))
    # Keep reference
    canvas.tk_img_ref = tk_img


def on_mousewheel(event):
    """Enable SHIFT+Scroll for horizontal, normal scroll for vertical,
    and CTRL+Scroll for zoom."""
    shift_down = (event.state & 0x0001) != 0
    ctrl_down = (event.state & 0x0004) != 0
    delta = event.delta

    # On Windows, event.delta is usually 120 or -120
    # On some systems, it might be different.

    if shift_down:
        # Horizontal scroll
        canvas.xview_scroll(int(-1*(delta/120)), "units")
    elif ctrl_down:
        # Zoom
        global scale_factor
        if delta > 0:
            scale_factor += 0.1
        else:
            scale_factor = max(0.1, scale_factor - 0.1)
        draw_boxes()
    else:
        # Vertical scroll
        canvas.yview_scroll(int(-1*(delta/120)), "units")

########################################################
# EDIT TEXT
########################################################

def edit_text(event):
    # Convert canvas coords -> actual image coords.
    cx = canvas.canvasx(event.x)
    cy = canvas.canvasy(event.y)

    x_orig = int(cx / scale_factor)
    y_orig = int(cy / scale_factor)

    for item in text_data:
        x_min, y_min = item["xy_a"]
        x_max, y_max = item["xy_b"]
        if x_min <= x_orig <= x_max and y_min <= y_orig <= y_max:
            new_text = simpledialog.askstring("Edit Text", f"Edit text [{item['index']}]:", initialvalue=item["text"])
            if new_text is not None:
                undo_stack.append([d.copy() for d in text_data])
                item["text"] = new_text
                # Mark confidence as uncertain if changed
                item["confidence"] = 0.5
                draw_boxes()
                display_text()
            break

########################################################
# DISPLAY TEXT
########################################################
text_box = None

def display_text():
    text_box.delete(1.0, tk.END)
    for item in text_data:
        text_box.insert(tk.END,
            f"[{item['index']}] {item['text']} (Confidence: {item['confidence']:.2f})\n")

########################################################
# UNDO/REDO
########################################################

def undo():
    global text_data
    if undo_stack:
        redo_stack.append([d.copy() for d in text_data])
        text_data = undo_stack.pop()
        draw_boxes()
        display_text()

def redo():
    global text_data
    if redo_stack:
        undo_stack.append([d.copy() for d in text_data])
        text_data = redo_stack.pop()
        draw_boxes()
        display_text()

########################################################
# SAVE TEXT
########################################################

def save_text():
    file_path = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV Files", "*.csv"), ("JSON Files", "*.json")]
    )
    if not file_path:
        return

    if file_path.endswith('.csv'):
        with open(file_path, "w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "text", "confidence", "xy_a", "xy_b"])
            for item in text_data:
                writer.writerow([
                    item["index"],
                    item["text"],
                    item["confidence"],
                    item["xy_a"],
                    item["xy_b"]
                ])
    elif file_path.endswith('.json'):
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(text_data, f, indent=4)

########################################################
# ZOOM CONTROLS
########################################################

def zoom_in():
    global scale_factor
    scale_factor += 0.1
    draw_boxes()

def zoom_out():
    global scale_factor
    scale_factor = max(0.1, scale_factor - 0.1)
    draw_boxes()

########################################################
# BUILD UI
########################################################

tk_root = tk.Tk()
tk_root.title("🐆 OCELOT - OCR Correction Tool")
tk_root.geometry("1200x900")

# Simple Theming via ttk.Style
style = ttk.Style(tk_root)
# Use 'clam' or other themes: ('clam', 'alt', 'default', 'classic')
style.theme_use("classic")

# For a LabelFrame, the internal style name is TLabelframe
style.configure("TLabelframe", background="#b8baf5")
style.configure("TLabelframe.Label", background="#b8baf5", foreground="white")

# For frames, “TFrame” is usually available
style.configure("TFrame", background="#b8baf5")
style.configure("TButton", padding=5)

# Top Frame
top_frame = ttk.Frame(tk_root, padding=5, style="TFrame")
top_frame.pack(side=tk.TOP, fill=tk.X)

load_btn = ttk.Button(top_frame, text="📂 Load Image", command=load_image, width=15)
load_btn.pack(side=tk.LEFT, padx=5)

webcam_btn = ttk.Button(top_frame, text="📸 Webcam", command=webcam_preview, width=15)
webcam_btn.pack(side=tk.LEFT, padx=5)

save_btn = ttk.Button(top_frame, text="💾 Save CSV/JSON", command=save_text, width=15)
save_btn.pack(side=tk.LEFT, padx=5)

undo_btn = ttk.Button(top_frame, text="↩️ Undo", command=undo, width=8)
undo_btn.pack(side=tk.LEFT, padx=5)

redo_btn = ttk.Button(top_frame, text="↪️ Redo", command=redo, width=8)
redo_btn.pack(side=tk.LEFT, padx=5)

zoom_in_btn = ttk.Button(top_frame, text="🔍+", command=zoom_in, width=5)
zoom_in_btn.pack(side=tk.LEFT, padx=5)

zoom_out_btn = ttk.Button(top_frame, text="🔍-", command=zoom_out, width=5)
zoom_out_btn.pack(side=tk.LEFT, padx=5)

# Canvas Frame (with scrollbars)
canvas_frame = ttk.Frame(tk_root, style="TFrame")
canvas_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

canvas_scroll_y = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL)
canvas_scroll_x = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)

canvas = tk.Canvas(canvas_frame, bg="#f0f0f0",
                   xscrollcommand=canvas_scroll_x.set,
                   yscrollcommand=canvas_scroll_y.set)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

canvas_scroll_y.config(command=canvas.yview)
canvas_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)

canvas_scroll_x.config(command=canvas.xview)
canvas_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)

# SHIFT+Scroll=horizontal, CTRL+Scroll=zoom, else vertical
canvas.bind_all("<MouseWheel>", on_mousewheel)
canvas.bind_all("<Button-4>", on_mousewheel)  # Linux might use Button-4/5
canvas.bind_all("<Button-5>", on_mousewheel)

# Click event for editing text
canvas.bind("<Button-1>", edit_text)

# Extracted Text Frame (corrected)
text_frame = ttk.LabelFrame(
    tk_root, 
    text="Extracted Text", 
    padding=5, 
    style="TLabelframe"  # Corrected from "TLabelFrame"
)
text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# Scrollable text box inside the frame
text_box = scrolledtext.ScrolledText(text_frame, wrap=tk.WORD, height=10)
text_box.pack(fill=tk.BOTH, expand=True)

# Mainloop
tk_root.mainloop()
