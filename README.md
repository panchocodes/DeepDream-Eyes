# Deepdream Eyes
Made in collaboration with Chris George for Art + ML Spring 2019.
Course Website: https://kangeunsu.com/artml19s/

## Final Results
### Original images are the `inital_#` images.
`layer = ‘mixed4d_3x3_bottleneck_pre_relu’`
`T(layer)[:,:,:,142] + T(layer)[:,:,:,8]`

![](./final_1.jpg "(1.2)")
![](./final_2.png "(2.2)")
![](./final_3.png "(3.2)")
![](./final_4.png "(4.2)")

## Code
#### deep_dream_edit.py
The code we wrote was only for easily running a photo on a specific layer and every channel in that layer. We modified the `render_deapdream` function so that it returned the image to be saved into the correct directory.
```
def render_deepdream(...)
    ...
    return PIL.Image.fromarray(np.uint8(np.clip(img/255.0, 0, 1)*255))

image_name = 'insert_image_path'
layer = 'mixed4d_3x3_bottleneck_pre_relu'
new_file_path = './'+layer+'/'
for i in range(1, 84):
    img0 = PIL.Image.open(image_name)
    img0 = np.float32(img0)
    deep_dream_image = render_deepdream(T(layer)[:,:,:,i], img0)
    deep_dream_image.save(new_file_path+str(i)+'.jpeg')
```
We included two of those tests in this repo. One was for `mized4d_3x3_bottleneck_pre_relu` and another was for `mixed4b_3x3_bottleneck_pre_relu`.

## Intermediate Results
We also had some intermediate results before we settled on eyes for our final project. Some of those results can be found in the intermediate results folder.
