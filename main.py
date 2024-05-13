from tkinter import *
from tkinter import filedialog

from ultralytics import YOLO
from ultralytics.solutions import heatmap
import cv2
import time
import datetime
from bin.VideoZone import detect_from_video_zone
from bin.photo_detect import photo_detect
from bin.photo_detect_all import photo_detect_all
from bin.video import detect_from_video
from bin.video_track import analyze_video
from face_detection.delete import get_employees_name
from face_detection.delete import show_records
from face_detection.delete import delete_record
from face_detection.add_to_base import add_to_base
import customtkinter
from CTkTable import *
from PIL import Image, ImageTk, ImageDraw

employees_name_array=get_employees_name()




def change_active_button(self,selection):
    if selection == 'rectangle':
        self.draw_polyline_button.configure(fg_color="#1F6AA5",hover_color="#144870")
        self.draw_rectangle_button.configure(fg_color="#146341",hover_color="#0b3623")

    elif selection =="polyline":
        self.draw_rectangle_button.configure(fg_color="#1F6AA5",hover_color="#144870")
        self.draw_polyline_button.configure(fg_color="#146341",hover_color="#0b3623")

def video_button_handler(method,place=""):
    if method == 'camera':
        if place == "krakau":
            model = "models/tokioKrakau6000.pt"
            detect_from_video("rtsp://localhost:8554/file?file=krakauStragan.mkv",model, centered="false")
        elif place == "tokio":
            model = "models/tokioKrakau5000.pt"
            detect_from_video("rtsp://localhost:8554/file?file=tokio.mkv",model,centered="true")
    elif method == "faces":
        model = "models/best_today.pt"
        detect_from_video_zone("rtsp://localhost:8554/file?file=ProjektMBox.mkv", model)
    else:
        model = "models/tokioKrakau5000.pt"
        try:
            filepath =filedialog.askopenfilename(filetypes=[("Videos", "*.mp4;*.avi;*.mkv;*.mov")])
        except:
            print("Error loading file")
        if(len(filepath)>0):
            toplevel_window = ToplevelWindow()
            toplevel_window.attributes('-topmost', 'true')
            toplevel_window.wait_window()
            if(not heatmap_choice):
                detect_from_video(filepath, model, centered="false")
            else:
                App.generate_heatmap(app,filepath,model)
customtkinter.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

width = 1280
height = 720
title="NazwaRobocza™"
model_path = "models/tokioKrakau5.pt"
heatmap_choice = False


class ToplevelWindow(customtkinter.CTkToplevel):
    def generate_heatmap(self):
        global heatmap_choice
        self.destroy()
        heatmap_choice = True
    def contiune_with_video_analysis(self):
        global heatmap_choice
        self.destroy()
        heatmap_choice = False
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry("400x300")
        self.grid_rowconfigure((0,1), weight=1)
        self.grid_columnconfigure((0,1), weight=1)

        self.label=customtkinter.CTkLabel(self, text="Czy generować mapę położenia? \n"
                                                     "Przy wyborze tej opcji nie będzie możliwy\n podgląd na żywo \n\n"
                                                     "Nie - analiza tłumu z podanego filmu")

        self.cancel_button=customtkinter.CTkButton(self, text="Nie", command=self.contiune_with_video_analysis)
        self.generate_heatmap_button=customtkinter.CTkButton(self, text="Generuj mapę", command=self.generate_heatmap)

        self.label.grid(column=0,row=0, columnspan=2, sticky="nsew")
        self.cancel_button.grid(row=1,column=0)
        self.generate_heatmap_button.grid(row=1,column=1)



class App(customtkinter.CTk):
    drawing_with_rectangle = False
    selected_areas=[]
    polyline_area=[]
    isDrawing = False
    employeePanelToggle = False
    employeeImagePath = ""

    def generate_heatmap(self, video_path, model):
        remaining_width = (self.winfo_width() - self.sidebar_frame.winfo_width()) - 300
        progressbar_lagel=  customtkinter.CTkLabel(app, text="Generowanie  mapy położenia \nProszę czekać...",  font=customtkinter.CTkFont(size=20, weight="bold"))
        franes_label = customtkinter.CTkLabel(app, text="",  font=customtkinter.CTkFont(size=20, weight="bold"))
        time_left_label = customtkinter.CTkLabel(app, text="", font=customtkinter.CTkFont(size=20, weight="bold"))
        progressbar = customtkinter.CTkProgressBar(app, orientation="horizontal", height=45, width=remaining_width,
                                                   progress_color="#7d15ed")
        progressbar.set(0)
        progressbar_lagel.grid(row=0, column=1, sticky="nsew")
        progressbar.grid(row=1, column=1)
        franes_label.grid(row=2, column=1, sticky="nsew", pady=0)
        time_left_label.grid(row=3, column=1, sticky="nsew", pady=0)

        model = YOLO(model)
        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        # Video writer
        video_writer = cv2.VideoWriter("heatmap_output.avi",
                                       cv2.VideoWriter_fourcc(*'mp4v'),
                                       fps,
                                       (w, h))
        frame_count = 0
        # Init heatmap
        heatmap_obj = heatmap.Heatmap()
        heatmap_obj.set_args(colormap=cv2.COLORMAP_MAGMA,
                             imw=w,
                             imh=h,
                             shape="circle",
                             classes_names=model.names,
                             decay_factor=1)

        while cap.isOpened():
            start=time.time()
            all_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            success, im0 = cap.read()
            if not success:
                print("Video frame is empty or video processing has been successfully completed.")
                break
            imgsz = (1088, 1920)
            tracks = model.track(im0, persist=True, imgsz=imgsz,
                                  augment=True, iou=0.01, max_det=10000)

            im0 = heatmap_obj.generate_heatmap(im0, tracks)
            video_writer.write(im0)
            progressbar.set(frame_count/all_frames)
            franes_label.configure(text=f"Klatka: {str(frame_count)}/{str(all_frames)}")
            progressbar.update_idletasks()
            frame_count += 1
            end=time.time()
            timer=end-start

            frame_remaining=all_frames - frame_count
            time_remaining=frame_remaining*timer
            time_proper = str(datetime.timedelta(seconds=int(time_remaining)))

            time_left_label.configure(text=f"Pozostały czas: {time_proper}")

            app.update()
        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()

        progressbar.grid_remove()
        franes_label.grid_remove()
        progressbar_lagel.grid_remove()
        time_left_label.grid_remove()
    def add_employee(self):
        employee_name=self.add_employee_entry.get().strip()
        if(len(employee_name)==0):
            self.add_employee_entry.delete(0, 'end')
            self.add_employee_entry.configure(placeholder_text="Pole nie może być puste!")
            self.focus()
        else:
            employees_name_array.append(employee_name)
            self.employees_dropdown.configure(values=employees_name_array)
            self.employees_dropdown.set(employee_name)
            self.get_employee_records(employee_name)
    def delete_from_database(self,row):
        delete_record(self.table.get(row,0))
    def get_employee_photo(self):
        try:
            self.employeeImagePath = filedialog.askopenfilename(filetypes=[("Images", "*.jpg;*.jpeg;*.png;*.webp")])
        except:
            print("Error loading employee photo")
        if len(self.employeeImagePath)>0:
            self.employee_image_direction_dropdown.configure(state="normal")
            self.employee_image_direction_dropdown.set("up")
        
    def add_employee_to_database(self,choice):
        print("Zdjęcie: "+self.employeeImagePath)
        print("Pracownik: "+self.employees_dropdown.get())
        print("Kierunek: "+choice)

        add_to_base(self.employees_dropdown.get(),self.employeeImagePath,choice)
        self.get_employee_records(self.employees_dropdown.get())

    def delete_from_table(self,index, choice):
        delete_record(index)
        self.get_employee_records(choice)

    def get_employee_records(self, choice):
        employee_records = show_records(choice)
        if hasattr(self, "table"):
            self.table.grid_remove()
        self.table_frame = customtkinter.CTkScrollableFrame(self)
        self.table_frame.grid_rowconfigure(0, weight=1)
        self.table_frame.grid_columnconfigure(0, weight=1)
        self.table_frame.grid(column=1, row=0, sticky="nsew", rowspan=11)

        self.table = CTkTable(master=self.table_frame,values=employee_records)
        self.table.grid(column=0,row=0, pady=(20,0))

        self.table.add_column(["Usuń"],3)
        for i in range(1, len(self.table.get())):
            self.table.insert(i,3,"[USUŃ]")
            self.table.edit(i,3, command=lambda index=i : self.delete_from_table(self.table.get(index, 0),choice))
    def toggle_employee_panel(self):
        if self.employeePanelToggle:
            self.employees_button.configure(text="Panel pracowników")
            self.employees_button.configure( image=customtkinter.CTkImage(
                dark_image=Image.open("sprites/user_icon.png")))
            self.analysis_source_frame.grid(row=2, column=0, rowspan=3, sticky="new")
            self.analysis_source_frame.grid(row=2, column=0, rowspan=3, sticky="new")

            if(hasattr(self,"table_frame")):
                self.table_frame.grid_remove()
            if hasattr(self,"add_employee_button_frame"):
                self.add_employee_button_frame.grid_remove()
            self.employees_frame.grid_remove()
            self.employee_action_frame.grid_remove()
            self.employee_image_direction_frame.grid_remove()
        else:
            self.employees_button.configure(text="Powrót do analizy")
            self.employees_button.configure( image=customtkinter.CTkImage(
                dark_image=Image.open("sprites/back_arrow_left.png")))
            self.analysis_source_frame.grid_remove()
            if hasattr(self,"canvas"):
                self.canvas.grid_remove()
                self.drawing_method_frame.grid_remove()
            self.employees_frame.grid(row=2, column=0, rowspan=4, sticky="new")
            self.employee_action_frame.grid(row=6, column=0, rowspan=2, sticky="new")
            self.employee_image_direction_frame.grid(row=8, column=0, rowspan=2, sticky="new")
            self.add_employee_button_frame.grid(row=10, column=0, sticky="new")
            self.get_employee_records(self.employees_dropdown.get())
        self.employeePanelToggle = not self.employeePanelToggle

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
    def analyze_photos(self,image_path,selected_areas=[],process_whole_frame=False,fragments=False):
        model = "models/28_best.pt"
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        # process whole frame -> photodetect all
        # not process whole frame ->
        if fragments:
            analyzed_image = photo_detect(image_path, model, selected_areas, process_whole_frame,  width=canvas_width, height=canvas_height)
        elif process_whole_frame:
             analyzed_image = photo_detect_all(image_path, model, width=canvas_width, height=canvas_height)
        else:
            analyzed_image = photo_detect(image_path,model,selected_areas,  width=canvas_width, height=canvas_height)
        self.canvas.image = analyzed_image
        self.canvas.delete("all")
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
                if (len(self.selected_areas) > 0):
                    self.analyze_button.configure(state="normal")

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
        if len(self.filepath) <= 0:
            return
        self.drawing_method_frame.grid(row=5, column=0, rowspan=3, sticky="n")
        sidebar_frame_width=self.sidebar_frame.cget("width")
        remaining_width=self.winfo_width()-sidebar_frame_width
        actual_height = self.winfo_height()
        self.image = Image.open(self.filepath)
        #self.image = self.image.resize((remaining_width, actual_height))
        self.draw = ImageDraw.Draw(self.image)
        self.photo = ImageTk.PhotoImage(image=self.image)
        self.canvas = Canvas(self)
        self.canvas.grid(row=0, rowspan=11,columnspan=2, column=1, sticky="nsew")
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
        self.grid_rowconfigure((0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10), weight=1)
        # create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=11, sticky="nsew")
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text=title, font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        #==================================DRAWING METHODS + FRAME==================================

        self.drawing_method_frame = customtkinter.CTkFrame(self.sidebar_frame, corner_radius=0)

        self.drawing_method_frame.columnconfigure(2, weight=1)
        self.drawing_method_frame.grid_rowconfigure(4, weight=1)
        self.drawing_method_title=customtkinter.CTkLabel(self.drawing_method_frame,text="Wybierz metodę\nzaznaczania:",
                                                         font=customtkinter.CTkFont(size=20, weight="bold"))

        self.draw_rectangle_button = customtkinter.CTkButton(self.drawing_method_frame, image=customtkinter.CTkImage(
            dark_image=Image.open("sprites/draw_rectangle.png")),
                                                             width=15,text="",command=self.select_with_rectangle)
        self.draw_polyline_button = customtkinter.CTkButton(self.drawing_method_frame, image=customtkinter.CTkImage(
            dark_image=Image.open("sprites/draw_freely.png")),width=15,text="",
                                                             command=self.select_with_polyline)
        self.analyze_button = customtkinter.CTkButton(self.drawing_method_frame, text="Analizuj zaznaczony obszar",
                                                      state="disabled",
                                                      command=lambda: self.analyze_photos(self.filepath,
                                                                                          self.selected_areas))
        self.analyze_whole_button = customtkinter.CTkButton(self.drawing_method_frame, text="Analizuj całe zdjęcie",
                                                      command=lambda: self.analyze_photos(self.filepath,
                                                                                          self.selected_areas,
                                                                                          process_whole_frame=True))

        self.analyze_fragments_button = customtkinter.CTkButton(self.drawing_method_frame, text="Analizuj zdjęcie fragmentami",
                                                      command=lambda: self.analyze_photos(self.filepath,
                                                                                          self.selected_areas,
                                                                                          process_whole_frame=True,  fragments=True))

        self.drawing_method_title.grid(row=0, column=0, pady=10, columnspan=2)
        self.draw_rectangle_button.grid(row=1, column=0,columnspan=1, pady=10)
        self.draw_polyline_button.grid(row=1, column=1,columnspan=1, pady=10)
        self.analyze_button.grid(row=2, column=0, columnspan=2, pady=10)
        self.analyze_whole_button.grid(row=3, column=0, columnspan=2, pady=10)
        self.analyze_fragments_button.grid(row=4, column=0, columnspan=2, pady=10)

        # ==================================END OF DRAWING METHODS + FRAME==================================

        #self.drawing_method_frame.grid_remove()
        # ================================== EMPLOYEES FRANE =========================================

        self.employees_button = customtkinter.CTkButton(self.sidebar_frame,text="Panel pracowników",width=20,
                                                         image=customtkinter.CTkImage(
                                                             dark_image=Image.open("sprites/user_icon.png"))
                                                           ,command=self.toggle_employee_panel)
        self.employees_button.grid(row=1, column=0, pady=10, padx=0)

        self.employees_frame=customtkinter.CTkFrame(self.sidebar_frame, corner_radius=0)


        self.employees_dropdown_label=customtkinter.CTkLabel(self.employees_frame,
                                                          text="Wybierz pracownika:",
                                                          font=customtkinter.CTkFont(size=20, weight="bold"))

        self.employees_dropdown=customtkinter.CTkComboBox(master=self.employees_frame,
                                     values=employees_name_array,
                                     command=self.get_employee_records)

        self.add_employee_button = customtkinter.CTkButton(self.employees_frame,text="Dodaj pracownika",width=20,
                                                           command=self.add_employee,fg_color="#198754", hover_color="#0f4f31")

        self.add_employee_entry = customtkinter.CTkEntry(self.employees_frame, placeholder_text="Podaj nazwę pracownika",insertwidth=3)

        self.add_employee_label=customtkinter.CTkLabel(self.employees_frame,
                                                          text="Dodaj pracownika:",
                                                          font=customtkinter.CTkFont(size=20, weight="bold"))

        self.employees_dropdown_label.grid(row=0, column=0, pady=10)
        self.employees_dropdown.grid(row=1, column=0, pady=10, padx=10)
        self.add_employee_label.grid(row=3, column=0, pady=10, padx=10)
        self.add_employee_entry.grid(row=4, column=0,columnspan=2, pady=10, padx=20,sticky="new")
        self.add_employee_button.grid(row=5, column=0,columnspan=2, pady=10, padx=20,sticky="new")
        # ==================================END OF EMPLOYEES FRANE==================================

        # ================================== EMPLOYEE ACTION FRANE =================================
        self.employee_action_frame = customtkinter.CTkFrame(self.sidebar_frame, corner_radius=0)

        self.employee_action_frame.grid_rowconfigure((0,1), weight=1)
        self.employee_action_frame.grid_columnconfigure((0), weight=1)

        self.employee_action_label=customtkinter.CTkLabel(self.employee_action_frame,
                                                          text="Wybierz akcję:",
                                                          font=customtkinter.CTkFont(size=20, weight="bold"))

        self.add_image_button = customtkinter.CTkButton(self.employee_action_frame,text="Dodaj zdjęcie",width=20,
                                                         image=customtkinter.CTkImage(
                                                             dark_image=Image.open("sprites/add_image.png"))
                                                           ,command=self.get_employee_photo)
        self.employee_action_label.grid(row=0, column=0,columnspan=2, pady=10, padx=20,sticky="new")
        self.add_image_button.grid(row=1, column=0,columnspan=2, pady=10, padx=20,sticky="new")

        self.employee_image_direction_frame=customtkinter.CTkFrame(self.sidebar_frame, corner_radius=0)
        self.employee_image_direction_frame.grid_rowconfigure((0,1), weight=1)
        self.employee_image_direction_frame.grid_columnconfigure(0, weight=1)

        self.add_employee_button_frame=customtkinter.CTkFrame(self.sidebar_frame, corner_radius=0)
        self.add_employee_button_frame.grid_rowconfigure(0, weight=1)
        self.add_employee_button_frame.grid_columnconfigure(0, weight=1)

        self.add_employee_button = customtkinter.CTkButton(self.add_employee_button_frame, text="Dodaj",
                                                           command=lambda: self.add_employee_to_database
                                                           (self.employee_image_direction_dropdown.get()))

        self.add_employee_button.grid(row=0,column=0, pady=20)


        self.employee_image_direction_label=customtkinter.CTkLabel(self.employee_image_direction_frame,
                                                          text="Wybierz kierunek:",
                                                          font=customtkinter.CTkFont(size=20, weight="bold"))
        self.employee_image_direction_dropdown=customtkinter.CTkComboBox(self.employee_image_direction_frame,
                                                                         values=["up", "down","left","right"],
                                                                         state="disabled")
        self.employee_image_direction_label.grid(row=0, column=0)
        self.employee_image_direction_dropdown.grid(row=1,column=0)


        # ==================================END OF EMPLOYEE ACTION FRANE============================
        # ================================== ANALYSIS SOURCE FRANE =========================================

        self.analysis_source_frame = customtkinter.CTkFrame(self.sidebar_frame, corner_radius=0)
        self.analysis_source_frame.grid(row=2, column=0, rowspan=3, sticky="new")
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
                                                                ,command=lambda: video_button_handler(""))

        self.camera_krakau_button = customtkinter.CTkButton(self.analysis_source_frame, text="Kraków",width=20,
                                                          image=customtkinter.CTkImage(
                                                              dark_image=Image.open("sprites/from_camera.png"))
                                                                ,command=lambda:video_button_handler("camera",'krakau'))

        self.camera_tokio_button = customtkinter.CTkButton(self.analysis_source_frame, text="Tokio",width=20,
                                                          image=customtkinter.CTkImage(
                                                              dark_image=Image.open("sprites/from_camera.png"))
                                                                ,command=lambda:video_button_handler("camera",'tokio'))

        self.analyze_faces_button = customtkinter.CTkButton(self.analysis_source_frame, text="Wykrywanie twarzy \ni liczenie czynności",
                                                          width=20,
                                                          image=customtkinter.CTkImage(
                                                              dark_image=Image.open("sprites/from_camera.png"))
                                                                ,command=lambda:video_button_handler("faces"))

        self.analysis_source_label.grid(row=0, column=0, pady=10, columnspan=2)
        self.from_photo_button.grid(row=1, column=0, pady=10,padx=10)
        self.from_video_button.grid(row=1, column=1, pady=10,padx=10)
        self.camera_tokio_button.grid(row=2, column=0, pady=10,padx=10)
        self.camera_krakau_button.grid(row=2, column=1, pady=10,padx=10)
        self.analyze_faces_button.grid(row=3, column=0, columnspan=2, pady=5,padx=10)
        # ================================== END OF ANALYSIS SOURCE FRANE ==================================



        # self.choose_photo_button.grid(row=3, column=0, padx=20, pady=10)
        # self.analyze_button.grid(row=4, column=0, padx=20, pady=10)
        # self.detect_from_video_button.grid(row=5, column=0, padx=20, pady=10)
        # self.detect_from_camera_button.grid(row=6, column=0, padx=20, pady=10)


if __name__ == "__main__":
    app=App()
    app.mainloop()

