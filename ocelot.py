"""
OCELOT - OCR Correction Tool
Features in this build:
1) Bounding Box Resizing
2) Handwriting Recognition Toggle
3) Auto-detect file type: image vs audio vs (video?)
4) If bounding box is resized, we prompt user to re-run OCR on that region
5) SHIFT+Scroll=horizontal, CTRL+Scroll=zoom, normal=vertical
6) Undo/Redo, CSV/JSON export
7) (Optional) speech_recognition for audio
"""

import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog, scrolledtext, simpledialog, messagebox
import os
import cv2
from PIL import Image, ImageTk
import easyocr
import csv
import json
import math

# Optional: for speech recognition
try:
    import speech_recognition as sr
    HAVE_SPEECH = True
except ImportError:
    HAVE_SPEECH = False

# === OCR READERS ===
reader_standard = easyocr.Reader(["en"])   # normal
reader_handwriting = easyocr.Reader(["en"]) # placeholder for handwriting

use_handwriting = False  # Toggle

# === Globals ===
img = None
zoom_img = None
text_data = []
undo_stack = []
redo_stack = []
scale_factor = 1.0

dragging_corner = None
dragging_box_index = None

tk_root = None
canvas = None
canvas_scroll_x = None
canvas_scroll_y = None
text_box = None

# --- FILE DETECTION HELPERS ---
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}

def load_file():
    """
    Prompt user to load a file. If it's an image, do OCR.
    If it's audio, do STT. If it's video or unsupported, show warning.
    """
    try:
        file_path = filedialog.askopenfilename(
            filetypes=[("All Files", "*.*")]
        )
        if not file_path:
            return

        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        if ext in IMAGE_EXTENSIONS:
            load_image(file_path)
        elif ext in AUDIO_EXTENSIONS:
            if HAVE_SPEECH:
                audio_to_text(file_path)
            else:
                messagebox.showerror("Error", "Speech libraries not installed. Try pip install SpeechRecognition pyaudio.")
        elif ext in VIDEO_EXTENSIONS:
            messagebox.showinfo("Info", f"Video loading not implemented yet.\nFile: {file_path}")
        else:
            messagebox.showinfo("Info", f"Unsupported extension: {ext}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load file: {e}")

def load_image(file_path):
    global img, scale_factor
    scale_factor = 1.0
    try:
        img_bgr = cv2.imread(file_path)
        if img_bgr is None:
            raise ValueError("Failed to load image. The file may be corrupted or in an unsupported format.")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = img_rgb
        process_image(img)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load image: {e}")

def audio_to_text(file_path):
    """
    Simple speech-to-text using speech_recognition on an audio file.
    """
    r = sr.Recognizer()
    try:
        with sr.AudioFile(file_path) as source:
            audio_data = r.record(source)
        text_result = r.recognize_google(audio_data)
        messagebox.showinfo("STT Result", f"Transcribed Text:\n{text_result}")
    except sr.UnknownValueError:
        messagebox.showwarning("Speech Recognition", "Could not understand audio.")
    except sr.RequestError as e:
        messagebox.showerror("Speech Recognition", f"Error: {e}")

def process_image(image):
    global text_data
    undo_stack.clear()
    redo_stack.clear()

    if use_handwriting:
        results = reader_handwriting.readtext(image, detail=True)
    else:
        results = reader_standard.readtext(image, detail=True)

    text_data = []
    for bbox, text, confidence in results:
        xy_a = tuple(map(int, bbox[0]))
        xy_b = tuple(map(int, bbox[2]))
        text_data.append({
            "index": len(text_data),
            "text": text,
            "confidence": confidence,
            "xy_a": xy_a,
            "xy_b": xy_b
        })

    draw_boxes()
    display_text()

def draw_boxes():
    global zoom_img, img, scale_factor
    if img is None:
        return

    w = int(img.shape[1] * scale_factor)
    h = int(img.shape[0] * scale_factor)
    zoom_img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    display_img = zoom_img.copy()

    for item in text_data:
        idx = item["index"]
        x_min, y_min = item["xy_a"]
        x_max, y_max = item["xy_b"]
        confidence = item["confidence"]

        x_min_s = int(x_min * scale_factor)
        y_min_s = int(y_min * scale_factor)
        x_max_s = int(x_max * scale_factor)
        y_max_s = int(y_max * scale_factor)

        # Color by confidence
        if confidence > 0.85:
            color = (0,255,0)
        elif confidence < 0.5:
            color = (0,0,255)
        else:
            color = (255,255,0)

        cv2.rectangle(display_img, (x_min_s,y_min_s), (x_max_s,y_max_s), color, 2)
        cv2.putText(display_img, f"{idx}", (x_min_s, y_min_s - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Corner squares
        corner_size = 6
        corners = [
            (x_min_s, y_min_s),
            (x_max_s, y_min_s),
            (x_min_s, y_max_s),
            (x_max_s, y_max_s),
        ]
        for (cx, cy) in corners:
            cv2.rectangle(display_img, (cx - corner_size, cy - corner_size),
                                     (cx + corner_size, cy + corner_size),
                                     color, cv2.FILLED)

    pil_img = Image.fromarray(display_img)
    tk_img = ImageTk.PhotoImage(pil_img)

    canvas.delete("all")
    canvas.create_image(0,0, anchor=tk.NW, image=tk_img)
    canvas.config(scrollregion=canvas.bbox(tk.ALL))
    canvas.tk_img_ref = tk_img

def on_mousewheel(event):
    shift_down = (event.state & 0x0001) != 0
    ctrl_down = (event.state & 0x0004) != 0
    delta = event.delta

    if shift_down:
        canvas.xview_scroll(int(-1 * (delta/120)), "units")
    elif ctrl_down:
        global scale_factor
        if delta > 0:
            scale_factor += 0.1
        else:
            scale_factor = max(0.1, scale_factor - 0.1)
        draw_boxes()
    else:
        canvas.yview_scroll(int(-1 * (delta/120)), "units")

def on_mouse_down(event):
    global dragging_corner, dragging_box_index
    cx = canvas.canvasx(event.x)
    cy = canvas.canvasy(event.y)

    x_img = int(cx / scale_factor)
    y_img = int(cy / scale_factor)

    # Check for ctrl => text edit
    ctrl_down = (event.state & 0x0004) != 0
    if ctrl_down:
        box_edit_click(x_img,y_img)
        return

    # else corner drag
    corner_hit_range = 10
    found_corner = None
    found_box_idx = None

    for item in text_data:
        idx = item["index"]
        x_min, y_min = item["xy_a"]
        x_max, y_max = item["xy_b"]
        corners = [
            ("tl", (x_min, y_min)),
            ("tr", (x_max, y_min)),
            ("bl", (x_min, y_max)),
            ("br", (x_max, y_max)),
        ]
        for corner_label, (cxn,cyn) in corners:
            dist = math.hypot(x_img - cxn, y_img - cyn)
            if dist <= corner_hit_range:
                found_corner = corner_label
                found_box_idx = idx
                break
        if found_corner:
            break

    if found_corner:
        dragging_corner = found_corner
        dragging_box_index = found_box_idx
    else:
        dragging_corner = None
        dragging_box_index = None

def on_mouse_move(event):
    global dragging_corner, dragging_box_index
    if dragging_corner is None or dragging_box_index is None:
        return

    cx = canvas.canvasx(event.x)
    cy = canvas.canvasy(event.y)

    x_img = int(cx / scale_factor)
    y_img = int(cy / scale_factor)

    for item in text_data:
        if item["index"] == dragging_box_index:
            x_min,y_min = item["xy_a"]
            x_max,y_max = item["xy_b"]

            if dragging_corner == "tl":
                x_min = min(x_max, x_img)
                y_min = min(y_max, y_img)
            elif dragging_corner == "tr":
                x_max = max(x_min, x_img)
                y_min = min(y_max, y_img)
            elif dragging_corner == "bl":
                x_min = min(x_max, x_img)
                y_max = max(y_min, y_img)
            elif dragging_corner == "br":
                x_max = max(x_min, x_img)
                y_max = max(y_min, y_img)

            item["xy_a"] = (x_min,y_min)
            item["xy_b"] = (x_max,y_max)
            break
    draw_boxes()

def on_mouse_up(event):
    global dragging_corner, dragging_box_index
    if dragging_corner is not None and dragging_box_index is not None:
        # Prompt user if they want to re-OCR the updated bounding box
        ans = messagebox.askyesno("Re-OCR?",
              "Bounding box was resized.\nRe-run OCR on this region?")
        if ans:
            # Re-OCR just that region
            # We'll extract that region from the original image & run OCR
            for item in text_data:
                if item["index"] == dragging_box_index:
                    x_min,y_min = item["xy_a"]
                    x_max,y_max = item["xy_b"]
                    if x_min < 0: x_min=0
                    if y_min < 0: y_min=0
                    if x_max>img.shape[1]: x_max=img.shape[1]
                    if y_max>img.shape[0]: y_max=img.shape[0]
                    sub_img = img[y_min:y_max, x_min:x_max]
                    if use_handwriting:
                        sub_results = reader_handwriting.readtext(sub_img, detail=False)
                    else:
                        sub_results = reader_standard.readtext(sub_img, detail=False)

                    if sub_results:
                        # Let's just take the first line from sub_results
                        new_text = " ".join(sub_results)
                        # record old text for undo
                        undo_stack.append([d.copy() for d in text_data])
                        item["text"] = new_text
                        # mark confidence as 0.9
                        item["confidence"] = 0.9
                    draw_boxes()
                    display_text()
                    break

    dragging_corner = None
    dragging_box_index = None

def box_edit_click(x_img,y_img):
    # Ctrl + click => text edit
    for item in text_data:
        x_min,y_min = item["xy_a"]
        x_max,y_max = item["xy_b"]
        if x_min<=x_img<=x_max and y_min<=y_img<=y_max:
            new_text = simpledialog.askstring("Edit Text", f"Edit text [{item['index']}]:",
                            initialvalue=item["text"])
            if new_text is not None:
                undo_stack.append([d.copy() for d in text_data])
                item["text"] = new_text
                item["confidence"] = 0.5
                draw_boxes()
                display_text()
            return

def display_text():
    text_box.delete(1.0, tk.END)
    for item in text_data:
        text_box.insert(tk.END,
            f"[{item['index']}] {item['text']} (Confidence: {item['confidence']:.2f})\n")

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

def save_text():
    file_path = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV Files", "*.csv"),("JSON Files","*.json")]
    )
    if not file_path:
        return

    if file_path.endswith(".csv"):
        with open(file_path,"w",newline="",encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["index","text","confidence","xy_a","xy_b"])
            for item in text_data:
                writer.writerow([
                    item["index"],
                    item["text"],
                    item["confidence"],
                    item["xy_a"],
                    item["xy_b"]
                ])
    elif file_path.endswith(".json"):
        with open(file_path,"w",encoding="utf-8") as f:
            json.dump(text_data,f,indent=4)

def zoom_in():
    global scale_factor
    scale_factor += 0.1
    draw_boxes()

def zoom_out():
    global scale_factor
    scale_factor = max(0.1, scale_factor-0.1)
    draw_boxes()

def toggle_handwriting():
    global use_handwriting
    use_handwriting = not use_handwriting
    msg = "Handwriting mode ON" if use_handwriting else "Handwriting mode OFF"
    messagebox.showinfo("OCR Mode", msg)

def build_ui():
    global tk_root, canvas, canvas_scroll_x, canvas_scroll_y, text_box
    tk_root = tk.Tk()
    tk_root.title("🐆 OCELOT - OCR Correction Tool")
    tk_root.geometry("1200x900")

    style = ttk.Style(tk_root)
    style.theme_use("clam")

    top_frame = ttk.Frame(tk_root, padding=5)
    top_frame.pack(side=tk.TOP, fill=tk.X)

    ttk.Button(top_frame, text="📂 Load File", command=load_file, width=15).pack(side=tk.LEFT, padx=5)
    ttk.Button(top_frame, text="💾 Save CSV/JSON", command=save_text, width=15).pack(side=tk.LEFT, padx=5)
    ttk.Button(top_frame, text="↩️ Undo", command=undo, width=8).pack(side=tk.LEFT, padx=5)
    ttk.Button(top_frame, text="↪️ Redo", command=redo, width=8).pack(side=tk.LEFT, padx=5)
    ttk.Button(top_frame, text="🔍+", command=zoom_in, width=5).pack(side=tk.LEFT, padx=5)
    ttk.Button(top_frame, text="🔍-", command=zoom_out, width=5).pack(side=tk.LEFT, padx=5)
    ttk.Button(top_frame, text="✒️ Toggle Handwriting", command=toggle_handwriting).pack(side=tk.LEFT, padx=5)

    canvas_frame = ttk.Frame(tk_root)
    canvas_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    canvas_scroll_y = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL)
    canvas_scroll_x = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)

    c = tk.Canvas(canvas_frame, bg="#f0f0f0",
                  xscrollcommand=canvas_scroll_x.set,
                  yscrollcommand=canvas_scroll_y.set)
    c.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    canvas_scroll_y.config(command=c.yview)
    canvas_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)

    canvas_scroll_x.config(command=c.xview)
    canvas_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)

    # SHIFT+Scroll=horizontal, CTRL+Scroll=zoom, else vertical
    c.bind_all("<MouseWheel>", on_mousewheel)
    c.bind_all("<Button-4>", on_mousewheel)  # Linux
    c.bind_all("<Button-5>", on_mousewheel)

    # bounding box resizing
    c.bind("<Button-1>", on_mouse_down)
    c.bind("<B1-Motion>", on_mouse_move)
    c.bind("<ButtonRelease-1>", on_mouse_up)

    global canvas
    canvas = c

    # text area
    text_frame = ttk.LabelFrame(tk_root, text="Extracted Text", padding=5)
    text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    st = scrolledtext.ScrolledText(text_frame, wrap=tk.WORD, height=10)
    st.pack(fill=tk.BOTH, expand=True)

    global text_box
    text_box = st

    tk_root.mainloop()

if __name__ == "__main__":
    build_ui()
