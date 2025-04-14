"""
Ocelot Kivy Overhaul (Single File)
Features:
- Load an image with EasyOCR
- Display in a Scatter for zoom/pan
- Draw bounding boxes over recognized text
- Click inside a bounding box to edit text
- Drag corners to resize bounding box
- [Placeholder for partial re-OCR if needed]
"""

import math
import cv2
import numpy as np
from PIL import Image as PILImage
import os

import easyocr

import kivy
kivy.require("2.1.0")  # or whichever version you use

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.scatter import Scatter
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserIconView, FileChooserListView
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.properties import NumericProperty, ListProperty, StringProperty, BooleanProperty
from kivy.graphics import Color, Rectangle, Line, InstructionGroup
from kivy.uix.label import Label as KivyLabel
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.spinner import Spinner
from kivy.core.window import Window

import csv
import json
import copy
import tkinter as tk
from tkinter import filedialog

# ============ OCR Readers ============
reader_standard = easyocr.Reader(["en"], gpu=False) # Standard reader, force CPU
# Placeholder for a specific handwriting model if you have one
# reader_handwriting = easyocr.Reader(['en'], recog_network='your_handwriting_model', gpu=False)
reader_handwriting = easyocr.Reader(["en"], gpu=False) # Using standard for now, force CPU

reader = reader_standard # Default reader
use_handwriting = False # Flag to toggle

# ============ Data Structures ============
class OCRBox:
    """
    Simple container for bounding box data recognized by EasyOCR
    xy_a: (x_min, y_min) - Top-Left
    xy_b: (x_max, y_max) - Bottom-Right
    text: recognized text
    conf: confidence
    """
    def __init__(self, xy_a, xy_b, text, conf):
        self.xy_a = xy_a
        self.xy_b = xy_b
        self.text = text
        self.conf = conf

    # Add a simple way to create a copy
    def copy(self):
        return OCRBox(self.xy_a, self.xy_b, self.text, self.conf)

# ============ Custom Widgets ============
class ImageViewer(Scatter):
    """
    Displays the loaded image, handles mouse interactions for zoom, pan, resize, edit, and drawing.
    Drawings are done on the canvas.
    """
    boxes = ListProperty([])
    img_np = None
    img_w = NumericProperty(0)
    img_h = NumericProperty(0)
    
    # State variables for interactions
    dragging_corner = None
    dragging_box = None
    drawing_box = BooleanProperty(False) # True when drawing a new box
    draw_start_pos = ListProperty([0, 0]) # Start position for new box drawing
    panning = BooleanProperty(False) # True when panning with middle mouse
    last_pan_pos = ListProperty([0, 0]) # Last position during panning

    def __init__(self, **kwargs):
        super().__init__(
            do_rotation=False, 
            do_scale=False, # Disable default scaling, we handle via scroll
            do_translation=False, # Disable default translation, we handle via middle mouse
            scale_min=0.1, 
            scale_max=5.0, 
            **kwargs
        )
        self.image = Image(allow_stretch=True, keep_ratio=True)
        self.add_widget(self.image)
        self.box_graphics = InstructionGroup()
        self.canvas.after.add(self.box_graphics)
        self.bind(boxes=self.redraw_boxes)
        self.register_event_type("on_box_modified")
        self.bind(scale=self.redraw_boxes) 
        self.double_tap_time = 0
        self.last_touch_pos = None
        
        # Add a temporary graphic instruction for drawing the new box
        self.draw_rect_instruction = None

    def update_image(self, img_np):
        """Convert np array -> Kivy texture, display it."""
        if img_np is None:
            self.image.texture = None
            self.img_np = None
            self.img_h, self.img_w = 0, 0
            self.image.size = (0, 0)
            self.boxes = []
            return

        self.img_np = img_np
        h, w, _ = img_np.shape
        self.img_h, self.img_w = h, w

        # Convert BGR (OpenCV default) or RGB to texture
        # Assuming input is RGB from PIL
        buf = cv2.flip(img_np, 0).tobytes() # Flip vertical for Kivy buffer
        tex = Texture.create(size=(w,h), colorfmt='rgb')
        tex.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
        # No need to flip texture itself if buffer was flipped
        # tex.flip_vertical()

        self.image.texture = tex
        self.image.size = (w, h) # Set image widget size to texture size
        self.size = (w, h)       # Set Scatter size to image size initially

        # Reset scale and position of the Scatter container
        self.scale = 1.0
        self.pos = (0, 0)

        # No need to explicitly call redraw_boxes, binding handles it when boxes are set/cleared
        # self.redraw_boxes() # Implicitly called if self.boxes is changed

    # --- Coordinate Conversion ---
    def to_image_space(self, x, y):
        """Convert scatter window coords -> image coords (int)."""
        # Use scatter's to_local which accounts for scale and translation
        lx, ly = self.to_local(x, y, relative=False) # relative=False for window coords

        # Kivy's y=0 is bottom, Image's y=0 is top (after texture flip)
        # However, graphics instructions use Kivy coords (y=0 bottom)
        # And our box coords (xy_a, xy_b) are image coords (y=0 top)
        # We need consistency. Let's keep image coords (y=0 top) internally
        # and convert only for drawing/interaction.

        # For interaction (touch -> image space):
        # to_local gives coords relative to scatter's bottom-left.
        # Image is at (0,0) within scatter, same size.
        # So, lx, ly are coords relative to image bottom-left.
        # Convert ly to top-left origin: img_y = self.img_h - ly
        img_x = int(lx)
        #img_y = int(self.img_h - ly) # This caused issues, Kivy graphics work bottom-up
        img_y = int(ly) # Keep Kivy's bottom-up for graphics interaction

        # Clamp coords to image bounds just in case
        img_x = max(0, min(self.img_w, img_x))
        img_y = max(0, min(self.img_h, img_y))

        return img_x, img_y

    # --- Drawing ---
    def redraw_boxes(self, *args):
        """Clear and redraw existing boxes and the new box being drawn."""
        self.box_graphics.clear()
        if self.img_h == 0:
             return

        base_corner_size_px = 8
        corner_size = max(2.0, base_corner_size_px / self.scale)

        with self.box_graphics:
            # Draw existing boxes
            if self.boxes:
                for box in self.boxes:
                    x_min, y_min_img = box.xy_a
                    x_max, y_max_img = box.xy_b
                    y_min_kivy = self.img_h - y_max_img
                    y_max_kivy = self.img_h - y_min_img
                    
                    is_dragging_this_box = (self.dragging_box is box)
                    conf = box.conf
                    line_width = 2.5 if is_dragging_this_box else 1.5

                    if is_dragging_this_box:
                        Color(1, 0, 1, 1) # Magenta
                    elif conf > 0.85: Color(0, 1, 0, 1) # Green
                    elif conf > 0.5: Color(1, 1, 0, 1) # Yellow
                    else: Color(1, 0, 0, 1) # Red

                    Line(points=[x_min, y_min_kivy, x_max, y_min_kivy, x_max, y_max_kivy, x_min, y_max_kivy], width=line_width, close=True)
                    
                    corners_kivy = [(x_min, y_min_kivy), (x_max, y_min_kivy), (x_max, y_max_kivy), (x_min, y_max_kivy)]
                    for cx, cy in corners_kivy:
                        Rectangle(pos=(cx - corner_size / 2, cy - corner_size / 2), size=(corner_size, corner_size))
                        
            # Draw the new box being actively drawn (if any)
            if self.drawing_box and self.draw_rect_instruction:
                Color(0, 0.8, 1, 0.8) # Cyan for drawing new box
                Line(rectangle=self.draw_rect_instruction, width=1.5)

    # --- Mouse/Touch Handling --- 
    def on_touch_down(self, touch):
        if not self.collide_point(*touch.pos):
            return False 
            
        # --- Double Tap for Reset (Keep this) ---
        current_time = touch.time_start
        if self.last_touch_pos:
            delta_time = current_time - self.double_tap_time
            delta_x = abs(touch.x - self.last_touch_pos[0])
            delta_y = abs(touch.y - self.last_touch_pos[1])
            if delta_time < 0.3 and delta_x < 30 and delta_y < 30:
                self.scale = 1.0
                self.pos = (0, 0)
                self.double_tap_time = 0
                self.last_touch_pos = None
                app = App.get_running_app()
                if app: app.log("View reset via double-tap.")
                return True # Consume touch
        self.double_tap_time = current_time
        self.last_touch_pos = touch.pos
        # --- End Double Tap ---

        # Check if touch is *inside* the actual image texture area
        img_touch_pos = self.image.to_widget(*touch.pos, relative=True)
        if not self.image.collide_point(*img_touch_pos):
             # Allow interaction outside image (e.g. for panning if background is visible)
             # But only if it's middle mouse button for panning
             if touch.button == 'middle':
                 self.panning = True
                 self.last_pan_pos = list(touch.pos)
                 touch.grab(self)
                 return True
             return False # Ignore other clicks outside image

        img_x, img_y = self.to_image_space(*touch.pos)
        img_y_top = self.img_h - img_y
        corner_hit_range_px = 20 
        corner_hit_range_img = corner_hit_range_px / self.scale

        # === Interaction Logic ===
        if touch.button == 'left':
            # 1. Check for corner drag first
            for box in self.boxes:
                x_min, y_min_img = box.xy_a
                x_max, y_max_img = box.xy_b
                corners_img = [("tl", (x_min, y_min_img)), ("tr", (x_max, y_min_img)), ("bl", (x_min, y_max_img)), ("br", (x_max, y_max_img))]
                for label, (cx, cy) in corners_img:
                    dist = math.hypot(img_x - cx, img_y_top - cy)
                    if dist <= corner_hit_range_img:
                        self.dragging_corner = label
                        self.dragging_box = box
                        touch.grab(self)
                        self.redraw_boxes()
                        return True

            # 2. Check for click inside existing box for editing
            for box in self.boxes:
                x_min, y_min_img = box.xy_a
                x_max, y_max_img = box.xy_b
                if x_min <= img_x <= x_max and y_min_img <= img_y_top <= y_max_img:
                    self.edit_box_text(box)
                    return True # Consume touch, don't start drawing
            
            # 3. If not corner drag or edit, start drawing a new box
            self.drawing_box = True
            self.draw_start_pos = [img_x, img_y] # Use Kivy coords (y=0 bottom)
            # Initialize the drawing rectangle graphic (x, y, w, h)
            self.draw_rect_instruction = [img_x, img_y, 0, 0] 
            touch.grab(self)
            self.redraw_boxes() # Show the initial (tiny) rect
            return True

        elif touch.button == 'middle':
            # Start panning
            self.panning = True
            self.last_pan_pos = list(touch.pos)
            touch.grab(self)
            return True
            
        elif touch.button == 'scrollup':
            # Zoom In
            self.apply_zoom(1.1, touch.pos)
            return True
            
        elif touch.button == 'scrolldown':
            # Zoom Out
            self.apply_zoom(1 / 1.1, touch.pos)
            return True

        # Allow Scatter's default handling for other buttons if needed (e.g., right-click context menu)
        # return super().on_touch_down(touch) # Careful: this might re-enable default pan/zoom
        return False # Consume other buttons for now

    def on_touch_move(self, touch):
        if touch.grab_current is not self:
            return False # Only handle grabs we initiated

        if self.dragging_corner and self.dragging_box:
            # --- Corner Drag Logic (same as before) ---
            ix, iy = self.to_image_space(*touch.pos)
            iy_top = self.img_h - iy
            box = self.dragging_box
            x_min, y_min = box.xy_a
            x_max, y_max = box.xy_b
            if self.dragging_corner == "tl": x_min, y_min = min(x_max - 1, ix), min(y_max - 1, iy_top)
            elif self.dragging_corner == "tr": x_max, y_min = max(x_min + 1, ix), min(y_max - 1, iy_top)
            elif self.dragging_corner == "bl": x_min, y_max = min(x_max - 1, ix), max(y_min + 1, iy_top)
            elif self.dragging_corner == "br": x_max, y_max = max(x_min + 1, ix), max(y_min + 1, iy_top)
            box.xy_a = (int(max(0, x_min)), int(max(0, y_min)))
            box.xy_b = (int(min(self.img_w, x_max)), int(min(self.img_h, y_max)))
            self.redraw_boxes()
            return True
            # --- End Corner Drag ---
            
        elif self.drawing_box:
            # --- Update New Box Drawing --- 
            curr_img_x, curr_img_y = self.to_image_space(*touch.pos)
            start_x, start_y = self.draw_start_pos
            
            # Update the draw_rect_instruction [x, y, w, h]
            rect_x = min(start_x, curr_img_x)
            rect_y = min(start_y, curr_img_y)
            rect_w = abs(start_x - curr_img_x)
            rect_h = abs(start_y - curr_img_y)
            self.draw_rect_instruction = [int(rect_x), int(rect_y), int(rect_w), int(rect_h)]
            self.redraw_boxes() # Update visual feedback
            return True
            # --- End New Box Drawing ---
            
        elif self.panning:
            # --- Panning Logic --- 
            dx = touch.x - self.last_pan_pos[0]
            dy = touch.y - self.last_pan_pos[1]
            # Scatter's position is its bottom-left corner relative to parent
            # We need to adjust self.pos (which is Scatter's pos)
            self.pos = (self.pos[0] + dx, self.pos[1] + dy)
            self.last_pan_pos = list(touch.pos)
            return True
            # --- End Panning --- 

        return False # Don't pass to super if we grabbed

    def on_touch_up(self, touch):
        if touch.grab_current is self:
            touch.ungrab(self)
            resized_box = None
            newly_drawn_box = None

            if self.dragging_corner and self.dragging_box:
                resized_box = self.dragging_box
                self.dispatch("on_box_modified", self.dragging_box)
                # Log state change for undo *after* modification is complete
                app = App.get_running_app()
                if app: app.push_undo_state() 

            elif self.drawing_box:
                # --- Finalize New Box --- 
                if self.draw_rect_instruction and self.draw_rect_instruction[2] > 5 and self.draw_rect_instruction[3] > 5:
                    # Convert Kivy rect coords [x, y, w, h] (y=0 bottom) to OCRBox coords (y=0 top)
                    x, y, w, h = self.draw_rect_instruction
                    x_min = x
                    x_max = x + w
                    y_min_img = self.img_h - (y + h) # Top Y in image coords
                    y_max_img = self.img_h - y      # Bottom Y in image coords
                    
                    # Ensure valid ordering
                    x_min, x_max = min(x_min, x_max), max(x_min, x_max)
                    y_min_img, y_max_img = min(y_min_img, y_max_img), max(y_min_img, y_max_img)
                    
                    # Clamp to image bounds
                    x_min = int(max(0, x_min))
                    y_min_img = int(max(0, y_min_img))
                    x_max = int(min(self.img_w, x_max))
                    y_max_img = int(min(self.img_h, y_max_img))
                    
                    # Create the new OCRBox object
                    new_box = OCRBox(xy_a=(x_min, y_min_img), xy_b=(x_max, y_max_img), text="?", conf=0.0)
                    self.boxes.append(new_box) # Add to the list (triggers redraw)
                    newly_drawn_box = new_box # Mark for OCR
                    
                    # Log state change for undo *after* adding the box
                    app = App.get_running_app()
                    if app: app.push_undo_state()
                    
                self.drawing_box = False
                self.draw_rect_instruction = None
                # --- End Finalize New Box ---

            # Reset states
            current_dragging_box = self.dragging_box # Store ref before clearing
            self.dragging_corner = None
            self.dragging_box = None
            self.panning = False
            
            # Redraw to remove highlight/drawing rect
            if current_dragging_box or self.drawing_box == False: # Redraw if drag ended or draw finished
                 self.redraw_boxes()

            # Run partial OCR if needed
            box_to_ocr = resized_box if resized_box else newly_drawn_box
            if box_to_ocr:
                self.partial_ocr(box_to_ocr)
                self.redraw_boxes() # Redraw again after OCR updates text/conf

            return True

        # Reset panning state if touch up occurred without a grab
        self.panning = False 
        return False # Don't pass to super if we grabbed
        
    # --- Zoom Helper --- 
    def apply_zoom(self, factor, touch_pos):
        """Applies zoom centered around the touch position."""
        if not self.image.texture: return
        
        # Calculate zoom point in scatter coords
        local_pos = self.to_local(*touch_pos)
        
        # Store old scale and position
        old_scale = self.scale
        old_pos = self.pos
        
        # Calculate new scale
        new_scale = old_scale * factor
        new_scale = max(self.scale_min, min(self.scale_max, new_scale))
        if abs(new_scale - old_scale) < 1e-6: # Avoid tiny changes
            return
            
        self.scale = new_scale
        
        # Adjust position to keep the zoom point stationary under the cursor
        # Formula: new_pos = touch_pos - (local_pos - old_pos) * (new_scale / old_scale)
        # Simplified: new_pos = touch_pos - local_pos * new_scale + old_pos * new_scale
        # Scatter pos is bottom-left, adjust accordingly
        scale_ratio = new_scale / old_scale
        new_x = touch_pos[0] - local_pos[0] * scale_ratio
        new_y = touch_pos[1] - local_pos[1] * scale_ratio
        self.pos = (new_x, new_y)
        
        app = App.get_running_app()
        if app: app.log(f"Zoom: {self.scale:.2f}x")

    # --- Partial OCR --- 
    def partial_ocr(self, box):
        """Runs OCR on the region defined by the given box."""
        if self.img_np is None:
            print("Partial OCR Skipped: No image loaded.")
            return

        # Get box coordinates (image space, y=0 at top)
        x_min, y_min = box.xy_a
        x_max, y_max = box.xy_b

        # Ensure coordinates are valid integers and within bounds
        x_min = max(0, int(x_min))
        y_min = max(0, int(y_min))
        x_max = min(self.img_w, int(x_max))
        y_max = min(self.img_h, int(y_max))

        # Check if the box has valid dimensions
        if x_max <= x_min or y_max <= y_min:
            print(f"Skipping partial OCR: Invalid box dimensions ({x_min},{y_min})->({x_max},{y_max})")
            return

        # Extract the sub-image (Region of Interest - ROI)
        roi = self.img_np[y_min:y_max, x_min:x_max]

        if roi.size == 0:
            print("Skipping partial OCR: ROI is empty")
            return

        print(f"Running partial OCR on region: ({x_min},{y_min}) -> ({x_max},{y_max})")
        try:
            current_reader = reader_handwriting if use_handwriting else reader_standard
            sub_results = current_reader.readtext(roi, detail=1)

            # --- UNDO INTEGRATION --- 
            old_text = box.text
            old_conf = box.conf
            new_text = old_text
            new_conf = old_conf
            data_changed = False # Flag to track if change occurred

            if sub_results:
                _bbox, res_text, res_conf = sub_results[0]
                # Only update if result is different
                if res_text != old_text or abs(res_conf - old_conf) > 0.01: # Check conf difference too
                    new_text = res_text
                    new_conf = res_conf
                    data_changed = True
                    print(f"Partial OCR result: '{new_text}' (Conf: {new_conf:.2f})")
                else:
                    print("Partial OCR result matches existing data.")
            else:
                print("Partial OCR found no text in the region.")
                # Optional: Update to empty if desired
                # if old_text != "": 
                #    new_text = ""
                #    new_conf = 0.0
                #    data_changed = True 

            # If data actually changed, push undo state *before* applying the change
            if data_changed:
                app = App.get_running_app()
                if app:
                    app.push_undo_state() # Save state before modifying box
                box.text = new_text
                box.conf = new_conf
                # Redraw will be handled by the caller (on_touch_up)
            # --- END UNDO INTEGRATION ---

        except Exception as e:
            print(f"Error during partial OCR: {e}")

    def edit_box_text(self, box):
        """Enhanced popup to edit recognized text, integrates with undo."""
        from kivy.uix.label import Label as PopupLabel
        from kivy.uix.popup import Popup
        from kivy.uix.boxlayout import BoxLayout
        from kivy.uix.textinput import TextInput
        from kivy.uix.button import Button
        from kivy.app import App

        # Create a more attractive layout
        layout = BoxLayout(orientation='vertical', spacing=10, padding=15)
        
        # Box coordinates and confidence info in the header
        x_min, y_min = box.xy_a
        x_max, y_max = box.xy_b
        w = x_max - x_min
        h = y_max - y_min
        
        # Choose color based on confidence
        conf_color = [0, 0.8, 0, 1] if box.conf > 0.85 else [0.8, 0.8, 0, 1] if box.conf > 0.5 else [0.8, 0, 0, 1]
        
        # Header with information
        header = BoxLayout(orientation='vertical', size_hint_y=None, height=80)
        
        title = PopupLabel(
            text="Edit OCR Text",
            font_size='18sp',
            bold=True,
            size_hint_y=None,
            height=30
        )
        header.add_widget(title)
        
        info1 = PopupLabel(
            text=f"Position: ({x_min}, {y_min}) to ({x_max}, {y_max})",
            font_size='14sp',
            size_hint_y=None,
            height=25
        )
        header.add_widget(info1)
        
        info2 = PopupLabel(
            text=f"Size: {w}×{h} px • Confidence: {box.conf:.2f}",
            font_size='14sp',
            color=conf_color,
            size_hint_y=None,
            height=25
        )
        header.add_widget(info2)
        
        layout.add_widget(header)
        
        # Add a separator
        separator = BoxLayout(size_hint_y=None, height=2)
        with separator.canvas:
            Color(0.7, 0.7, 0.7, 1)
            Rectangle(pos=(0, 0), size=(800, 2))
        layout.add_widget(separator)

        # Text input field with the current text
        txt = TextInput(
            text=box.text,
            multiline=True,
            font_size='16sp',
            size_hint_y=None,
            height=120,
            background_color=(0.95, 0.95, 0.95, 1),
            foreground_color=(0, 0, 0, 1)
        )
        layout.add_widget(txt)
        
        # Button bar
        buttons = BoxLayout(size_hint_y=None, height=50, spacing=20)
        
        cancel_btn = Button(
            text="Cancel",
            size_hint_x=0.5,
            background_color=(0.8, 0.2, 0.2, 1),
            font_size='16sp'
        )
        
        ok_btn = Button(
            text="Save Changes",
            size_hint_x=0.5,
            background_color=(0.2, 0.7, 0.2, 1),
            font_size='16sp'
        )
        
        buttons.add_widget(cancel_btn)
        buttons.add_widget(ok_btn)
        layout.add_widget(buttons)
        
        # Create the popup
        popup = Popup(
            title="Edit OCR Text",
            content=layout,
            size_hint=(0.8, None),
            height=300,
            auto_dismiss=False
        )

        # Button functions
        def confirm(instance):
            new_text = txt.text
            # Check if the text actually changed
            if new_text != box.text:
                # --- UNDO INTEGRATION --- 
                app = App.get_running_app()
                if app:
                    app.push_undo_state() # Save state before modifying box
                # --- END UNDO INTEGRATION ---

                old_text = box.text # Store old text for logging
                box.text = new_text
                box.conf = 0.5 # Mark as manually edited
                
                # It's generally better to dispatch modifications *before* dismissing
                self.dispatch("on_box_modified", box)
                popup.dismiss()
                self.redraw_boxes() # Redraw needed to update color based on new conf
                
                # Log change if needed
                if app:
                    app.log(f"Box text changed: '{old_text}' -> '{box.text}'")
                    
                    # Update the text in the results panel if it exists
                    if hasattr(app, 'text_results'):
                        # Find the text result with this text and update it
                        app.refresh_text_results()
            else:
                 # Text didn't change, just dismiss without saving state
                 popup.dismiss()
                 print("Edit cancelled or text unchanged.")

        def cancel(instance):
            popup.dismiss()

        # Bind button actions
        ok_btn.bind(on_release=confirm)
        cancel_btn.bind(on_release=cancel)

        popup.open()

    # Remove the old _draw_boxes method entirely
    # def _draw_boxes(self):
    #     """DEPRECATED: Convert our self.img_np to texture, then draw boxes.
    #        Actually we already have texture, so let's redraw with bounding boxes using OpenCV overlays.
    #     """
    #     pass # Now handled by redraw_boxes using Kivy graphics

    # --- Box Modified Event ---
    def on_box_modified(self, box):
        pass

# ============ Main Application ============
class OcelotApp(App):
    INPUT_DIR = "input" # Define input directory constant

    def build(self):
        self.title = "🐆 OCELOT - Kivy OCR Tool"
        self.load_popup = None
        self.save_popup = None
        self.undo_stack = [] # Initialize undo stack
        self.redo_stack = [] # Initialize redo stack
        
        # --- Keyboard Setup ---
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self, 'text')
        if self._keyboard.widget:
            # If it exists, this widget is a VKeyboard object which you can use
            # to change the keyboard layout.
            pass
        self._keyboard.bind(on_key_down=self._on_key_down)
        # ---------------------

        # Set window icon if platform supports it
        try:
            # from kivy.core.window import Window # <-- Removed redundant import
            # Window.icon = 'assets/ocelot_icon.png' 
            pass # Keep try-except in case setting icon fails
        except Exception as e:
            print(f"Could not set window icon: {e}")

        # --- Create ./input directory if it doesn't exist ---
        if not os.path.exists(self.INPUT_DIR):
            try:
                os.makedirs(self.INPUT_DIR)
                print(f"Created directory: {self.INPUT_DIR}")
            except OSError as e:
                print(f"Error creating directory {self.INPUT_DIR}: {e}")
                # Optionally handle error, maybe disable load/save?
        # ------------------------------------------------------

        # Main layout - vertical stack
        root = BoxLayout(orientation='vertical', spacing=5, padding=[5, 5, 5, 5]) # Added bottom padding
        
        # ==========================================================================
        # Top Bar - Stays the same
        # ==========================================================================
        top_bar = BoxLayout(size_hint_y=None, height=65, spacing=10, padding=[5, 5, 5, 5])
        file_buttons = BoxLayout(size_hint_x=0.35, spacing=5)
        edit_buttons = BoxLayout(size_hint_x=0.35, spacing=5)
        ocr_buttons = BoxLayout(size_hint_x=0.3, spacing=5)
        # ... (Button definitions remain the same as before) ...
        # File Operations Group
        load_btn = Button(
            text="Load",
            font_size='16sp',
            background_color=(0.2, 0.4, 0.8, 1),
            size_hint_x=0.5,
            bold=True
        )
        load_btn.bind(on_release=self.trigger_native_load)
        file_buttons.add_widget(load_btn)

        save_btn = Button(
            text="Save",
            font_size='16sp',
            background_color=(0.2, 0.6, 0.3, 1),
            size_hint_x=0.5,
            bold=True
        )
        save_btn.bind(on_release=self.trigger_native_save)
        file_buttons.add_widget(save_btn)
        
        # Edit Operation Group
        undo_btn = Button(
            text="Undo",
            font_size='16sp',
            background_color=(0.6, 0.5, 0.2, 1),
            size_hint_x=0.5
        )
        undo_btn.bind(on_release=lambda x: self.undo())
        edit_buttons.add_widget(undo_btn)

        redo_btn = Button(
            text="Redo",
            font_size='16sp',
            background_color=(0.6, 0.5, 0.2, 1),
            size_hint_x=0.5
        )
        redo_btn.bind(on_release=lambda x: self.redo())
        edit_buttons.add_widget(redo_btn)
        
        # OCR Operation Group
        ocr_btn = Button(
            text="OCR",
            font_size='16sp',
            background_color=(0.8, 0.3, 0.3, 1),
            bold=True
        )
        ocr_btn.bind(on_release=lambda x: self.run_ocr())
        ocr_buttons.add_widget(ocr_btn)
        
        self.handwriting_btn = Button(
            text="Handwriting: OFF",
            font_size='16sp',
            background_color=(0.5, 0.3, 0.6, 1)
        )
        self.handwriting_btn.bind(on_release=self.toggle_handwriting)
        ocr_buttons.add_widget(self.handwriting_btn)
        
        top_bar.add_widget(file_buttons)
        top_bar.add_widget(edit_buttons)
        top_bar.add_widget(ocr_buttons)
        root.add_widget(top_bar)
        
        # ==========================================================================
        # Image Viewer Area (Takes up most vertical space)
        # ==========================================================================
        img_container = BoxLayout(size_hint_y=0.6, orientation='vertical') # Adjusted size_hint_y
        
        # Zoom controls bar (above image)
        zoom_bar = BoxLayout(size_hint_y=None, height=40, spacing=5)
        zoom_in_btn = Button(text="+", size_hint_x=0.15, background_color=(0.4, 0.4, 0.4, 1))
        zoom_in_btn.bind(on_release=lambda x: self.zoom_image(1.25))
        zoom_bar.add_widget(zoom_in_btn)
        zoom_out_btn = Button(text="-", size_hint_x=0.15, background_color=(0.4, 0.4, 0.4, 1))
        zoom_out_btn.bind(on_release=lambda x: self.zoom_image(0.8))
        zoom_bar.add_widget(zoom_out_btn)
        reset_zoom_btn = Button(text="Reset View", size_hint_x=0.3, background_color=(0.4, 0.4, 0.4, 1))
        reset_zoom_btn.bind(on_release=lambda x: self.reset_view())
        zoom_bar.add_widget(reset_zoom_btn)
        clear_btn = Button(text="Clear Boxes", size_hint_x=0.4, background_color=(0.7, 0.2, 0.2, 1))
        clear_btn.bind(on_release=lambda x: self.clear_boxes())
        zoom_bar.add_widget(clear_btn)
        img_container.add_widget(zoom_bar)
        
        # Main image viewer
        self.image_viewer = ImageViewer()
        self.image_viewer.bind(on_box_modified=self.handle_box_modified)
        img_container.add_widget(self.image_viewer)
        root.add_widget(img_container)
        
        # ==========================================================================
        # Text Results Area (Below image viewer)
        # ==========================================================================
        text_container = BoxLayout(orientation='vertical', size_hint_y=0.2) # Adjusted size_hint_y
        
        # Header label for text results
        text_header = Label(
            text="OCR Results",
            size_hint_y=None,
            height=40,
            font_size='16sp',
            bold=True,
            color=(1, 1, 1, 1)
        )
        with text_header.canvas.before:
            Color(0.2, 0.2, 0.3, 1)
            self.text_header_rect = Rectangle(pos=text_header.pos, size=text_header.size)
        def update_text_header_rect(instance, value):
            self.text_header_rect.pos = instance.pos
            self.text_header_rect.size = instance.size
        text_header.bind(pos=update_text_header_rect, size=update_text_header_rect)
        text_container.add_widget(text_header)
        
        # Scrollable text display for OCR results
        self.text_scroll = ScrollView()
        self.text_results = GridLayout(cols=1, size_hint_y=None, spacing=5, padding=5)
        self.text_results.bind(minimum_height=self.text_results.setter('height'))
        self.text_scroll.add_widget(self.text_results)
        text_container.add_widget(self.text_scroll)
        root.add_widget(text_container)
        
        # ==========================================================================
        # Console Log Area (At the bottom)
        # ==========================================================================
        log_header = Label(
            text="Console Log",
            size_hint_y=None,
            height=25,
            font_size='14sp',
            color=(0.8, 0.8, 0.8, 1)
        )
        root.add_widget(log_header)
        
        self.console = ScrollView(size_hint_y=0.1) # Adjusted size_hint_y
        self.console_grid = GridLayout(cols=1, size_hint_y=None, spacing=2, padding=2)
        self.console_grid.bind(minimum_height=self.console_grid.setter('height'))
        self.console.add_widget(self.console_grid)
        root.add_widget(self.console)
        
        # First time log message
        self.log("Welcome to OCELOT OCR Tool. Use scroll wheel to zoom, middle-drag to pan.")
        
        return root

    # --- Keyboard Handling --- 
    def _keyboard_closed(self):
        self._keyboard.unbind(on_key_down=self._on_key_down)
        self._keyboard = None

    def _on_key_down(self, keyboard, keycode, text, modifiers):
        # keycode is a tuple (code, string_representation)
        key_str = keycode[1]
        # modifiers is a list e.g. ['ctrl', 'shift']

        is_ctrl = 'ctrl' in modifiers

        if is_ctrl and key_str == 'z': # Ctrl + Z for Undo
            self.undo()
            return True # Mark key event as handled
        elif is_ctrl and key_str == 'y': # Ctrl + Y for Redo
            self.redo()
            return True # Mark key event as handled
        
        # Let other keys pass through
        return False

    # Add zoom helper methods
    def zoom_image(self, factor):
        """Zooms the image by the given factor."""
        if self.image_viewer:
            # Current scale * factor (e.g., 1.0 * 1.25 = 1.25x zoom in)
            new_scale = self.image_viewer.scale * factor
            # Limit zoom range to prevent extreme values
            new_scale = max(0.1, min(5.0, new_scale))
            self.image_viewer.scale = new_scale
            self.log(f"Zoom set to {new_scale:.2f}x")
    
    def reset_view(self):
        """Resets the view to original position and scale."""
        if self.image_viewer:
            self.image_viewer.scale = 1.0
            self.image_viewer.pos = (0, 0)
            self.log("View reset to original.")
            
    # Update run_ocr to also update text display area
    def run_ocr(self):
        """Use easyocr to detect bounding boxes, store them in image_viewer.boxes."""
        if self.image_viewer.img_np is None:
            self.log("No image loaded.")
            return

        # Save state *before* running OCR
        self.push_undo_state()

        self.log("Running OCR...")
        try:
            # Clear previous text results
            self.text_results.clear_widgets()
            
            # run easyocr
            # Important: EasyOCR expects BGR by default if using cv2 image path,
            # but works with RGB if given a numpy array directly. Our img_np is RGB.
            current_reader = reader_handwriting if use_handwriting else reader_standard
            results = current_reader.readtext(self.image_viewer.img_np, detail=True)

            boxes_data = []
            for i, (bbox, text, conf) in enumerate(results):
                # bbox format: [[tl_x, tl_y], [tr_x, tr_y], [br_x, br_y], [bl_x, bl_y]]
                # We need min/max x/y for our OCRBox format
                x_coords = [p[0] for p in bbox]
                y_coords = [p[1] for p in bbox]
                xy_a = (int(min(x_coords)), int(min(y_coords))) # Top-left
                xy_b = (int(max(x_coords)), int(max(y_coords))) # Bottom-right
                boxes_data.append(OCRBox(xy_a, xy_b, text, conf))
                
                # Add each text result to the text display area
                self.add_text_result(i, text, conf)

            # Setting this property triggers redraw_boxes via binding
            self.image_viewer.boxes = boxes_data
            self.log(f"OCR found {len(boxes_data)} text boxes.")

        except Exception as e:
             self.log(f"Error during OCR: {e}")
             
    def add_text_result(self, index, text, confidence):
        """Adds a text result to the text display area."""
        # Create a container for this text result
        result_container = BoxLayout(orientation='vertical', size_hint_y=None, height=60, padding=[5,5,5,5])
        
        # Create a header with index and confidence
        header = BoxLayout(size_hint_y=None, height=20)
        
        # Choose color based on confidence
        if confidence > 0.85:
            color = [0, 0.8, 0, 1]  # Green
            conf_text = "High"
        elif confidence > 0.5:
            color = [0.8, 0.8, 0, 1]  # Yellow
            conf_text = "Medium"
        else:
            color = [0.8, 0, 0, 1]  # Red
            conf_text = "Low"
            
        index_lbl = Label(
            text=f"#{index}",
            size_hint_x=0.2,
            color=color
        )
        header.add_widget(index_lbl)
        
        conf_lbl = Label(
            text=f"Conf: {conf_text} ({confidence:.2f})",
            size_hint_x=0.8,
            color=color
        )
        header.add_widget(conf_lbl)
        
        result_container.add_widget(header)
        
        # Create the main text content
        text_lbl = TextInput(
            text=text,
            size_hint_y=None,
            height=40,
            multiline=True,
            readonly=True  # Can be made editable if needed
        )
        result_container.add_widget(text_lbl)
        
        # Add a separator that resizes with the container
        with result_container.canvas.before:
            Color(0.3, 0.3, 0.3, 1)
            # Store the rectangle reference directly within the canvas instructions
            # No need to store on self, use lambdas to update the specific rect
            rect = Rectangle(pos=result_container.pos, size=(result_container.width, 1))
            # Use lambdas to ensure the correct rect instance is updated
            result_container.bind(
                pos=lambda instance, value: setattr(rect, 'pos', instance.pos),
                size=lambda instance, value: setattr(rect, 'size', (instance.width, 1))
            )
            
        self.text_results.add_widget(result_container)

    def show_load_dialog(self, instance):
        if self.load_popup: # Avoid multiple popups
            return
        content = LoadDialog(load_callback=self.load_image, cancel_callback=self.dismiss_load_popup)
        self.load_popup = Popup(title="Select Image File", content=content, size_hint=(0.9, 0.9))
        self.load_popup.open()

    def dismiss_load_popup(self):
        if self.load_popup:
            self.load_popup.dismiss()
            self.load_popup = None

    def load_image(self, path): # Modified to accept single path
        if not path or not isinstance(path, str):
             self.log(f"Invalid path provided for loading: {path}")
             return
             
        try:
            pil_image = PILImage.open(path).convert("RGB")
            img_np = np.array(pil_image)
            # Clear undo/redo history for new image
            self.undo_stack.clear()
            self.redo_stack.clear()
            self.image_viewer.update_image(img_np)
            # Clear the text results panel when loading a new image
            if hasattr(self, 'text_results'):
                self.text_results.clear_widgets()
            self.log(f"Loaded image: {os.path.basename(path)}. Undo history cleared.")
        except Exception as e:
            self.log(f"Error loading image: {e}")
            # Show an error popup to the user?
            popup = Popup(title='Image Load Error', 
                          content=Label(text=f'Failed to load:\n{os.path.basename(path)}\nError: {e}'),
                          size_hint=(0.6, 0.4))
            popup.open()
            self.image_viewer.update_image(None)
            if hasattr(self, 'text_results'): # Clear text results on error too
                self.text_results.clear_widgets()
            self.undo_stack.clear()
            self.redo_stack.clear()

    def show_save_dialog(self, instance):
        if self.save_popup:
            return
        if not self.image_viewer.boxes:
             self.log("No OCR data to save.")
             # Optionally show a Kivy info popup here
             return

        content = SaveDialog(save_callback=self.save_data, cancel_callback=self.dismiss_save_popup)
        self.save_popup = Popup(title="Save OCR Data", content=content, size_hint=(0.9, 0.9))
        self.save_popup.open()

    def dismiss_save_popup(self):
        if self.save_popup:
            self.save_popup.dismiss()
            self.save_popup = None

    def save_data(self, full_path): # Now accepts full path directly
        self.dismiss_save_popup()
        if not full_path:
            self.log("Save cancelled.")
            return

        boxes_to_save = self.image_viewer.boxes
        if not boxes_to_save:
            self.log("Nothing to save.")
            return

        # Prepare data (convert OCRBox objects to dictionaries for JSON/CSV)
        data_list = []
        for i, box in enumerate(boxes_to_save):
            data_list.append({
                'index': i, # Add an index for reference
                'text': box.text,
                'confidence': box.conf,
                'xy_a': box.xy_a, # Top-left (xmin, ymin_img)
                'xy_b': box.xy_b  # Bottom-right (xmax, ymax_img)
            })

        try:
            if full_path.lower().endswith(".csv"):
                with open(full_path, "w", newline="", encoding="utf-8") as f:
                    if not data_list:
                        f.write("index,text,confidence,xy_a,xy_b\n") # Write header even if empty
                        self.log(f"Saved empty CSV to: {os.path.basename(full_path)}")
                        return

                    writer = csv.DictWriter(f, fieldnames=data_list[0].keys())
                    writer.writeheader()
                    writer.writerows(data_list)
                self.log(f"Saved data as CSV: {os.path.basename(full_path)}")

            elif full_path.lower().endswith(".json"):
                with open(full_path, "w", encoding="utf-8") as f:
                    json.dump(data_list, f, indent=4)
                self.log(f"Saved data as JSON: {os.path.basename(full_path)}")
            else:
                self.log(f"Error: Unknown save format for {full_path}")

        except Exception as e:
            self.log(f"Error saving data to {full_path}: {e}")
            # Show error popup?
            # Popup(title='Save Error', content=Label(text=f'Failed to save file:\n{e}'), size_hint=(0.6, 0.3)).open()

    def clear_boxes(self):
        # Push undo state *before* clearing
        self.push_undo_state()
        
        # Clear boxes in viewer
        self.image_viewer.boxes = []
        
        # Clear the text results panel as well
        if hasattr(self, 'text_results'):
            self.text_results.clear_widgets()
            
        self.log("Cleared bounding boxes and text results.")

    def toggle_handwriting(self, instance):
        """Toggles the handwriting OCR mode."""
        global use_handwriting, reader
        use_handwriting = not use_handwriting
        reader = reader_handwriting if use_handwriting else reader_standard
        mode = "ON" if use_handwriting else "OFF"
        instance.text = f"Handwriting: {mode}"
        self.log(f"Handwriting mode set to {mode}. Re-run OCR if needed.")
        # Optional: Offer to re-run OCR with the new mode?
        # if self.image_viewer.img_np is not None:
        #    self.run_ocr() # Re-run automatically? Or prompt?

    def handle_box_modified(self, instance, box):
        """Handle when a box is modified (text edit or resize)."""
        # Ensure text display is up-to-date with box data
        # Note: Text edits already call refresh_text_results from the popup confirm
        # Resizing triggers partial_ocr, which updates the box, triggering this.
        # So, we need to refresh here primarily for resize-triggered conf/text changes.
        self.refresh_text_results()
        
        # This could also be used for other tasks like auto-save

    def refresh_text_results(self):
        """Refreshes the text results display to match current boxes data."""
        if not hasattr(self, 'image_viewer') or not hasattr(self, 'text_results'):
            return
            
        # Clear current results
        self.text_results.clear_widgets()
        
        # Repopulate with current data
        for i, box in enumerate(self.image_viewer.boxes):
            self.add_text_result(i, box.text, box.conf)
            
        self.log("Text results display refreshed.")

    def log(self, msg):
        # Use Kivy's Label for console messages
        from kivy.uix.label import Label as ConsoleLabel
        label = ConsoleLabel(text=str(msg), size_hint_y=None, height=30, halign='left', valign='top')
        label.bind(size=label.setter('text_size')) # For text wrapping
        self.console_grid.add_widget(label)
        # Scroll to bottom - schedule to allow layout update
        from kivy.clock import Clock
        Clock.schedule_once(lambda dt: setattr(self.console, 'scroll_y', 0), 0.1)

    # --- Undo/Redo State Management ---
    def push_undo_state(self):
        """Saves a deep copy of the current boxes state to the undo stack."""
        if self.image_viewer:
            # Create a deep copy of the boxes list and its contents
            current_state = [box.copy() for box in self.image_viewer.boxes]
            self.undo_stack.append(current_state)
            # Any new action clears the redo stack
            self.redo_stack.clear()
            self.log(f"Undo state saved ({len(self.undo_stack)} items)") # Debug log

    def undo(self):
        """Reverts to the previous state from the undo stack."""
        if not self.undo_stack:
            self.log("Nothing to undo.")
            return

        # Push current state to redo stack *before* changing it
        current_state = [box.copy() for box in self.image_viewer.boxes]
        self.redo_stack.append(current_state)

        # Pop the last state from undo and apply it
        last_state = self.undo_stack.pop()
        self.image_viewer.boxes = last_state # This triggers redraw via binding
        self.refresh_text_results() # Update text panel
        self.log(f"Undo successful. ({len(self.undo_stack)} items left)")

    def redo(self):
        """Re-applies the next state from the redo stack."""
        if not self.redo_stack:
            self.log("Nothing to redo.")
            return

        # Push current state back to undo stack *before* changing it
        current_state = [box.copy() for box in self.image_viewer.boxes]
        self.undo_stack.append(current_state)

        # Pop the state from redo and apply it
        next_state = self.redo_stack.pop()
        self.image_viewer.boxes = next_state # Triggers redraw
        self.refresh_text_results() # Update text panel
        self.log(f"Redo successful. ({len(self.redo_stack)} items left)")

    # ==========================================================================
    # Native File Dialog Methods
    # ==========================================================================
    def _get_tkinter_root(self):
        """Initializes and hides the Tkinter root window."""
        root = tk.Tk()
        root.withdraw() # Hide the main window
        return root

    def trigger_native_load(self, instance):
        """Opens the native OS file dialog to load an image."""
        root = self._get_tkinter_root()
        
        # Define file types for the dialog
        filetypes = [
            ("Image Files", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp"),
            ("All Files", "*.*"),
        ]
        
        # Get initial directory path (absolute)
        initial_dir_path = os.path.abspath(self.INPUT_DIR)

        try:
            file_path = filedialog.askopenfilename(
                parent=root, # Keep dialog on top
                title="Select an Image File",
                initialdir=initial_dir_path,
                filetypes=filetypes
            )
        except Exception as e:
             self.log(f"Error opening file dialog: {e}")
             file_path = None
        finally:
             root.destroy() # Destroy the hidden Tk window

        if file_path: # Check if a file was selected
            self.load_image(file_path) # Pass the single path string
        else:
            self.log("Image loading cancelled.")

    def trigger_native_save(self, instance):
        """Opens the native OS file dialog to save OCR data."""
        if not self.image_viewer.boxes:
            self.log("No OCR data to save.")
            # Optionally show a Kivy info popup here
            return
            
        root = self._get_tkinter_root()
        
        # Define file types for saving
        filetypes = [
            ("CSV Files", "*.csv"),
            ("JSON Files", "*.json"),
        ]
        
        # Get initial directory path (absolute)
        initial_dir_path = os.path.abspath(self.INPUT_DIR)

        try:
            save_path = filedialog.asksaveasfilename(
                parent=root,
                title="Save OCR Data As...",
                initialdir=initial_dir_path,
                defaultextension=".csv", # Default extension
                filetypes=filetypes
            )
        except Exception as e:
             self.log(f"Error opening save dialog: {e}")
             save_path = None
        finally:
             root.destroy()

        if save_path: # Check if a path was provided
            self.save_data(save_path)
        else:
            self.log("Save cancelled.")

if __name__ == "__main__":
    OcelotApp().run()
