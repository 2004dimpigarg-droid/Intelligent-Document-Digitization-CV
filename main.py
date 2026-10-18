"""Intelligent Document Digitization and Quality Enhancement System.
Designed for clear document photos with visible corners and a contrasting background.
"""
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk

TITLE = 'Intelligent Document Digitization'
PREVIEW_SIZE = (430, 500)


def order_points(points):
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    out = np.zeros((4, 2), dtype=np.float32)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    out[0] = pts[np.argmin(sums)]
    out[2] = pts[np.argmax(sums)]
    out[1] = pts[np.argmin(diffs)]
    out[3] = pts[np.argmax(diffs)]
    return out


def resize_for_detection(image, max_dimension=1200):
    h, w = image.shape[:2]
    scale = min(1.0, max_dimension / float(max(h, w)))
    if scale == 1.0:
        return image.copy(), scale
    small = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return small, scale


def find_document_corners(image):
    small, scale = resize_for_detection(image)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    min_area = small.shape[0] * small.shape[1] * 0.08
    for contour in contours[:40]:
        if cv2.contourArea(contour) < min_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return (order_points(approx.reshape(4, 2)) / scale).astype(np.float32)
    raise ValueError('Automatic page detection failed. Use a photo with all four corners visible and clear contrast between the paper and background.')


def correct_perspective(image, corners):
    tl, tr, br, bl = order_points(corners)
    width = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    height = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    if width < 50 or height < 50:
        raise ValueError('The detected page is too small to transform.')
    width, height = min(width, 5000), min(height, 7000)
    destination = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(order_points(corners), destination)
    return cv2.warpPerspective(image, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def enhance_document(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    enhanced = cv2.fastNlMeansDenoising(enhanced, None, h=5, templateWindowSize=7, searchWindowSize=21)
    binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11)
    return enhanced, binary


def cv_to_pil(image):
    if image.ndim == 2:
        return Image.fromarray(image)
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


class App:
    def __init__(self, root):
        self.root = root
        self.root.title(TITLE)
        self.root.geometry('1120x740')
        self.root.minsize(900, 600)
        self.path = None
        self.original = self.corrected = self.enhanced = self.binary = None
        self.photos = {}
        self._build_ui()

    def _build_ui(self):
        header = ttk.Frame(self.root, padding=(14, 12))
        header.pack(fill='x')
        ttk.Label(header, text=TITLE, font=('TkDefaultFont', 17, 'bold')).pack(anchor='w')
        ttk.Label(header, text='Detect a document page, correct perspective, enhance contrast, and save the result.').pack(anchor='w', pady=(4, 0))
        controls = ttk.Frame(self.root, padding=(14, 0, 14, 10))
        controls.pack(fill='x')
        ttk.Button(controls, text='1. Upload Image', command=self.upload).pack(side='left', padx=(0, 8))
        ttk.Button(controls, text='2. Detect + Correct + Enhance', command=self.process).pack(side='left', padx=(0, 8))
        ttk.Button(controls, text='3. Save Enhanced Result', command=self.save_enhanced).pack(side='left', padx=(0, 8))
        ttk.Button(controls, text='Save Black-and-White Scan', command=self.save_binary).pack(side='left')
        panels = ttk.Frame(self.root, padding=(14, 0, 14, 8))
        panels.pack(fill='both', expand=True)
        for col in range(3):
            panels.columnconfigure(col, weight=1, uniform='preview')
        panels.rowconfigure(1, weight=1)
        for col, (key, label) in enumerate([('original', 'Original'), ('corrected', 'Perspective-corrected'), ('enhanced', 'Enhanced grayscale')]):
            ttk.Label(panels, text=label, font=('TkDefaultFont', 11, 'bold')).grid(row=0, column=col, sticky='w', padx=5, pady=(0, 5))
            widget = ttk.Label(panels, text='Upload an image to begin', anchor='center', relief='groove')
            widget.grid(row=1, column=col, sticky='nsew', padx=5, pady=2)
            self.photos[key] = {'widget': widget, 'photo': None}
        footer = ttk.Frame(self.root, padding=(14, 0, 14, 12))
        footer.pack(fill='x')
        self.status = tk.StringVar(value='Ready. Choose a clear document photo with all four corners visible.')
        ttk.Label(footer, textvariable=self.status, wraplength=1050).pack(anchor='w')
        ttk.Label(footer, text='Tip: Start with a moderately tilted page on a darker, uncluttered surface. Black-and-white output is optional.', foreground='#555555', wraplength=1050).pack(anchor='w', pady=(4, 0))

    def show(self, key, image):
        widget = self.photos[key]['widget']
        if image is None:
            widget.configure(image='', text='No image yet')
            self.photos[key]['photo'] = None
            return
        pil = cv_to_pil(image)
        pil.thumbnail(PREVIEW_SIZE, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(pil)
        widget.configure(image=photo, text='')
        self.photos[key]['photo'] = photo

    def upload(self):
        path = filedialog.askopenfilename(title='Choose a document photograph', filetypes=[('Image files', '*.jpg *.jpeg *.png *.bmp *.tif *.tiff'), ('All files', '*.*')])
        if not path:
            return
        image = cv2.imread(path)
        if image is None:
            messagebox.showerror('Could not open image', 'The selected file could not be read as an image.')
            return
        self.path, self.original = path, image
        self.corrected = self.enhanced = self.binary = None
        self.show('original', image)
        self.show('corrected', None)
        self.show('enhanced', None)
        self.status.set(f'Loaded: {os.path.basename(path)}')

    def process(self):
        if self.original is None:
            messagebox.showinfo('Upload an image first', 'Choose a document photograph before processing.')
            return
        try:
            corners = find_document_corners(self.original)
            self.corrected = correct_perspective(self.original, corners)
            self.enhanced, self.binary = enhance_document(self.corrected)
            self.show('corrected', self.corrected)
            self.show('enhanced', self.enhanced)
            self.status.set('Processing complete. Review the previews and save the output you prefer.')
        except Exception as exc:
            messagebox.showerror('Processing failed', str(exc))
            self.status.set('Processing failed. Try a clearer photograph with visible page corners.')

    def save_image(self, image, label, suffix, extension='.png'):
        if image is None:
            messagebox.showinfo('Nothing to save', 'Upload and process an image first.')
            return
        stem = os.path.splitext(os.path.basename(self.path))[0] if self.path else 'document'
        path = filedialog.asksaveasfilename(title=f'Save {label}', defaultextension=extension, initialfile=f'{stem}_{suffix}{extension}', filetypes=[('PNG image', '*.png'), ('JPEG image', '*.jpg *.jpeg')])
        if not path:
            return
        if not cv2.imwrite(path, image):
            messagebox.showerror('Save failed', 'Could not save the image to the selected location.')
            return
        messagebox.showinfo('Saved', f'{label} saved to:\n{path}')
        self.status.set(f'Saved: {path}')

    def save_enhanced(self):
        self.save_image(self.enhanced, 'enhanced document', 'enhanced')

    def save_binary(self):
        self.save_image(self.binary, 'black-and-white scan', 'black_white')


def main():
    root = tk.Tk()
    try:
        style = ttk.Style()
        if 'clam' in style.theme_names():
            style.theme_use('clam')
    except tk.TclError:
        pass
    App(root)
    root.mainloop()


if __name__ == '__main__':
    main()
