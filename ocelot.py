# Required dependencies:
# pip install opencv-python pillow easyocr

import tkinter as tk
from tkinter import filedialog, scrolledtext, simpledialog, messagebox
import cv2
from PIL import Image, ImageTk
import easyocr

# Initialize EasyOCR Reader once to improve efficiency
reader = easyocr.Reader(["en"])

img = None
display_img = None
tk_img = None
text_data = []

def load_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")])
    if not file_path:
        return

    global img
    img = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
    process_image(img)

def capture_webcam():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        messagebox.showerror("Error", "Failed to capture image from webcam.")
        return

    global img
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    process_image(img)

def process_image(image):
    global text_data
    result = reader.readtext(image, detail=True)

    text_data = []
    for bbox, text, confidence in result:
        x_min, y_min = map(int, bbox[0])
        x_max, y_max = map(int, bbox[2])
        text_data.append((x_min, y_min, x_max, y_max, text, confidence))

    draw_boxes()
    display_text()

def draw_boxes():
    global display_img, tk_img, text_data
    display_img = img.copy()
    for idx, (x_min, y_min, x_max, y_max, _, confidence) in enumerate(text_data):
        color = (0, 255, 0) if confidence > 0.85 else (0, 0, 255) if confidence < 0.5 else (255, 255, 0)
        cv2.rectangle(display_img, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(display_img, f'{idx}', (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    pil_img = Image.fromarray(display_img)
    tk_img = ImageTk.PhotoImage(pil_img)
    canvas.delete("all")  # Clear canvas before drawing new image
    canvas.config(width=tk_img.width(), height=tk_img.height())
    canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)

def display_text():
    text_box.delete(1.0, tk.END)
    for idx, (_, _, _, _, text, confidence) in enumerate(text_data):
        text_box.insert(tk.END, f"[{idx}] {text} (Confidence: {confidence:.2f})\n")

def edit_text(event):
    x_click, y_click = event.x, event.y
    for idx, (x_min, y_min, x_max, y_max, text, _) in enumerate(text_data):
        if x_min <= x_click <= x_max and y_min <= y_click <= y_max:
            new_text = simpledialog.askstring("Edit Text", f"Edit text [{idx}]", initialvalue=text)
            if new_text is not None:
                text_data[idx] = (x_min, y_min, x_max, y_max, new_text, 1.0)
                draw_boxes()
                display_text()
            break

def save_text():
    file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt"), ("CSV Files", "*.csv"), ("JSON Files", "*.json")])
    if not file_path:
        return

    with open(file_path, "w", encoding="utf-8") as f:
        for _, _, _, _, text, _ in text_data:
            f.write(text + "\n")

tk_root = tk.Tk()
tk_root.title("🐆 OCELOT - OCR Correction Tool")
tk_root.geometry("900x700")

top_frame = tk.Frame(tk_root, padx=10, pady=10)
top_frame.pack(side=tk.TOP, fill=tk.X)

load_btn = tk.Button(top_frame, text="📂 Load Image", command=load_image, width=15)
load_btn.pack(side=tk.LEFT, padx=5)

webcam_btn = tk.Button(top_frame, text="📸 Webcam Capture", command=capture_webcam, width=15)
webcam_btn.pack(side=tk.LEFT, padx=5)

save_btn = tk.Button(top_frame, text="💾 Save Text", command=save_text, width=15)
save_btn.pack(side=tk.LEFT, padx=5)

canvas_frame = tk.Frame(tk_root, bd=2, relief=tk.SUNKEN)
canvas_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

canvas = tk.Canvas(canvas_frame, bg="#f0f0f0")
canvas.pack(fill=tk.BOTH, expand=True)
canvas.bind("<Button-1>", edit_text)

text_frame = tk.LabelFrame(tk_root, text="Extracted Text", padx=10, pady=10)
text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

text_box = scrolledtext.ScrolledText(text_frame, wrap=tk.WORD, height=10)
text_box.pack(fill=tk.BOTH, expand=True)

tk_root.mainloop()
