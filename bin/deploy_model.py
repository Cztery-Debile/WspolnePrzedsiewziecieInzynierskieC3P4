import ultralytics
from roboflow import Roboflow
rf = Roboflow(api_key="7Yv8AKotpzj4H2EhEwCv")
project = rf.workspace("testtesttest-pvzym").project("xddd-izdpx")
version = project.version(25)

version.deploy('yolov8',"../../models/",'tokioKrakau5.pt')