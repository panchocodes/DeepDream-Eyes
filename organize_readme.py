import os


directory_path = "./mixed4b_3x3_bottleneck_pre_relu/"

directory = os.listdir(directory_path)

new_read_me = directory_path + "README.md"
text = ""
i = 0
for file in directory:
    if (".jpeg" in file):
        i += 1
        text += '<p align="center">  <img src="'+str(i)+".jpeg"+'?"> </p>'
        text += '<p align="center">'+str(i)+".jpeg"+'</p>'
        text += "\n\n***\n\n"
        # text += '<p align="center"><img src="'+file+'?"></p>'
        # text += '<p align="center">'+file+'</p>'
        # text += "\n\n***\n\n"

print(text)
file = open(new_read_me, "w")
file.write(text)
