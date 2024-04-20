from tkinter import *
from tkinter import filedialog

from activitiy.New_model.VideoZone import detect_from_video_zone
from bin.photo_detect import photo_detect
from bin.video import detect_from_video
from activitiy.video_track import analyze_video
import customtkinter
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw

def change_active_button(self,selection):
    if selection == 'rectangle':
        self.draw_polyline_button.configure(fg_color="#1F6AA5",hover_color="#144870")
        self.draw_rectangle_button.configure(fg_color="#146341",hover_color="#0b3623")

    elif selection =="polyline":
        self.draw_rectangle_button.configure(fg_color="#1F6AA5",hover_color="#144870")
        self.draw_polyline_button.configure(fg_color="#146341",hover_color="#0b3623")

def video_button_handler(method):
    if method == 'camera':
        detect_from_video_zone("rtsp://localhost:8554/file?file=ProjektMBox.mkv",model_path)
    else:
        filepath = filedialog.askopenfilename(filetypes=[("Videos", "*.mp4;*.avi;*.mkv;*.mov")])
        analyze_video(model_path,filepath)

customtkinter.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

width = 1280
height = 720
title="NazwaRobocza™"
model_path = "models/best_today.pt"

class App(customtkinter.CTk):
    drawing_with_rectangle = False
    selected_areas=[]
    polyline_area=[]
    isDrawing = False


    def bind_selection(self, start_function, draw_function, end_function):
            self.canvas.bind("<Button-1>", start_function)
            self.canvas.bind("<B1-Motion>", draw_function)
            self.canvas.bind("<ButtonRelease-1>", end_function)
    def select_with_rectangle(self):
        self.bind_selection(self.start_rectangle,self.draw_rectangle,self.end_rectangle)
        change_active_button(self, "rectangle")
    def select_with_polyline(self):
        self.bind_selection(self.select_area,self.select_area,self.select_area)
        change_active_button(self,"polyline")
    def analyze_photos(self,image_path, model_path,selected_areas,process_whole_frame=False):
        analyzed_image = photo_detect(image_path, model_path,selected_areas,process_whole_frame)
        self.canvas.image = analyzed_image
        self.canvas.create_image(0, 0, image=analyzed_image, anchor=NW)
        self.selected_areas = []


    def select_area(self, event):
        x, y = event.x, event.y
        if event.type == '4':  # Button-1 pressed
            self.isDrawing = True
            self.polyline_area.append([(x, y)])
        elif event.type == '6':  # B1-Motion
            if self.isDrawing:
                if not self.polyline_area[0]: self.polyline_area.pop(0)
                self.polyline_area[-1].append((x, y))
                self.draw_highlighted_region()
        elif event.type == '5':  # ButtonRelease-1
            if self.isDrawing:
                self.polyline_area[-1].append((x, y))
                self.isDrawing = False
                self.selected_areas.append(self.polyline_area[0])
                self.polyline_area=[]

    # Function to draw highlighted regions
    def draw_highlighted_region(self):
        if self.polyline_area:
            for area in self.polyline_area:
                for i in range(len(area) - 1):
                    x0, y0 = area[i]
                    x1, y1 = area[i + 1]
                    self.canvas.create_line(x0, y0, x1, y1, fill="red", width=2)

    # Function to remove the last added area
    def remove_last_area(self):
        global selected_areas
        if selected_areas:
            selected_areas.pop()
            self.canvas.delete("all")
            self.draw_highlighted_region()

    def start_rectangle(self, event):
        # Record the starting point of the rectangle
        self.start_x, self.start_y = event.x, event.y

    def draw_rectangle(self, event):
        # Remove previous rectangle
        if self.rect_id:
            self.canvas.delete(self.rect_id)

        # Draw the rectangle dynamically
        self.rect_id = self.canvas.create_rectangle(self.start_x, self.start_y, event.x, event.y, outline="red",width=2)

    def end_rectangle(self, event):
        x1, y1, x2, y2 = self.start_x, self.start_y, event.x, event.y
        self.selected_areas.append((x1, y1, x2, y2))
        # Clear drawing variables
        self.start_x, self.start_y = None, None
        self.rect_id = None
        if(len(self.selected_areas)>0):
            self.analyze_button.configure(state="normal")
    def load_photo(self):
        self.filepath = filedialog.askopenfilename(filetypes=[("Images", "*.jpg;*.jpeg;*.png;*.webp")])
        self.drawing_method_frame.grid(row=5, column=0, rowspan=3, sticky="n")
        self.image = Image.open(self.filepath)

        self.draw = ImageDraw.Draw(self.image)
        self.photo = ImageTk.PhotoImage(image=self.image)
        self.canvas = Canvas(self)
        self.canvas.grid(row=0, rowspan=7,columnspan=2, column=1, sticky="nsew")
        self.canvas.create_image(0, 0, image=self.photo, anchor=NW)

        self.start_x, self.start_y = None, None
        self.rect_id = None


    def __init__(self):
        super().__init__()

        # configure window
        self.title("NazwaRobocza™ Recognition System")
        self.geometry(f"{1100}x{580}")
        self.analyze_activity = customtkinter.BooleanVar()
        self.analyze_faces = customtkinter.BooleanVar()
        # configure grid layout (4x4)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2, 3, 4, 5, 6), weight=1)

        # create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=7, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(6, weight=1)
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text=title, font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        #==================================DRAWING METHODS + FRAME==================================

        self.drawing_method_frame = customtkinter.CTkFrame(self.sidebar_frame, corner_radius=0)

        self.drawing_method_frame.columnconfigure(2, weight=1)
        self.drawing_method_frame.grid_rowconfigure(4, weight=1)
        self.drawing_method_title=customtkinter.CTkLabel(self.drawing_method_frame,text="Wybierz metodę\nzaznaczania:",
                                                         font=customtkinter.CTkFont(size=20, weight="bold"))

        self.draw_rectangle_button = customtkinter.CTkButton(self.drawing_method_frame, image=customtkinter.CTkImage(
            dark_image=Image.open("sprites/draw_rectangle.png")),text="Prostokąt",
                                                             width=15,command=self.select_with_rectangle)
        self.draw_polyline_button = customtkinter.CTkButton(self.drawing_method_frame, image=customtkinter.CTkImage(
            dark_image=Image.open("sprites/draw_freely.png")),text="Dowolne",width=15,
                                                             command=self.select_with_polyline)
        self.analyze_button = customtkinter.CTkButton(self.drawing_method_frame, text="Analizuj zaznaczony obszar",
                                                      state="disabled",
                                                      command=lambda: self.analyze_photos(self.filepath,
                                                                                          model_path,
                                                                                          self.selected_areas))
        self.analyze_whole_button = customtkinter.CTkButton(self.drawing_method_frame, text="Analizuj całe zdjęcie",
                                                      command=lambda: self.analyze_photos(self.filepath,
                                                                                          model_path,
                                                                                          self.selected_areas,process_whole_frame=True))

        self.drawing_method_title.grid(row=0, column=0, pady=10, columnspan=2)
        self.draw_rectangle_button.grid(row=1, column=0,columnspan=2, pady=10)
        self.draw_polyline_button.grid(row=2, column=0,columnspan=2, pady=10)
        self.analyze_button.grid(row=3, column=0, columnspan=2, pady=10)
        self.analyze_whole_button.grid(row=4, column=0, columnspan=2, pady=10)

        # ==================================END OF DRAWING METHODS + FRAME==================================

        #self.drawing_method_frame.grid_remove()
        # ================================== ANALYSIS SOURCE FRANE =========================================

        self.analysis_source_frame = customtkinter.CTkFrame(self.sidebar_frame, corner_radius=0)
        self.analysis_source_frame.grid(row=1, column=0, rowspan=3, sticky="new")
        self.analysis_source_frame.columnconfigure(2, weight=1)
        self.analysis_source_frame.grid_rowconfigure(3, weight=1)

        self.analysis_source_label=customtkinter.CTkLabel(self.analysis_source_frame,
                                                          text="Wybierz materiał\ndo analizy:",
                                                          font=customtkinter.CTkFont(size=20, weight="bold"))
        self.from_photo_button = customtkinter.CTkButton(self.analysis_source_frame,text="Zdjęcie",width=20,
                                                         image=customtkinter.CTkImage(
                                                             dark_image=Image.open("sprites/from_picture.png"))
                                                           ,command=self.load_photo)

        self.from_video_button = customtkinter.CTkButton(self.analysis_source_frame, text="Wideo",width=20,
                                                         image=customtkinter.CTkImage(
                                                             dark_image=Image.open("sprites/from_video.png"))
                                                                ,command=lambda:video_button_handler(""))

        self.from_camera_button = customtkinter.CTkButton(self.analysis_source_frame, text="Kamera",width=20,
                                                          image=customtkinter.CTkImage(
                                                              dark_image=Image.open("sprites/from_camera.png"))
                                                                ,command=lambda:video_button_handler("camera"))

        self.analysis_source_label.grid(row=0, column=0, pady=10, columnspan=2)
        self.from_photo_button.grid(row=1, column=0, pady=10,padx=10)
        self.from_video_button.grid(row=1, column=1, pady=10,padx=10)
        self.from_camera_button.grid(row=2, column=0,columnspan=2, pady=10,padx=30)
        # ================================== END OF ANALYSIS SOURCE FRANE ==================================



        # self.choose_photo_button.grid(row=3, column=0, padx=20, pady=10)
        # self.analyze_button.grid(row=4, column=0, padx=20, pady=10)
        # self.detect_from_video_button.grid(row=5, column=0, padx=20, pady=10)
        # self.detect_from_camera_button.grid(row=6, column=0, padx=20, pady=10)


if __name__ == "__main__":
    App().mainloop()

