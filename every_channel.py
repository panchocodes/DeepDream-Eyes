# Original render deepdream photo
def render_deepdream(t_obj, img0=img_noise,
                     iter_n=15, step=1.5, octave_n=15, octave_scale=1.4):
    t_score = tf.reduce_mean(t_obj) # defining the optimization objective
    t_grad = tf.gradients(t_score, t_input)[0] # behold the power of automatic differentiation!

    # split the image into a number of octaves getting smaller and smaller images
    img = img0
    octaves = []
    for i in range(octave_n-1):
        hw = img.shape[:2] #image height and width
        lo = resize(img, np.int32(np.float32(hw)/octave_scale)) #low frequency parts (smaller image)
        hi = img-resize(lo, hw) #high frequency parts (details)
        img = lo # next iteration rescale this one
        octaves.append(hi) # add the details to octaves

    # generate details octave by octave from samll image to large
    for octave in range(octave_n):
        if octave>0:
            hi = octaves[-octave]
            img = resize(img, hi.shape[:2])+hi
        for i in range(iter_n):
            g = calc_grad_tiled(img, t_grad)
            img += g*(step / (np.abs(g).mean()+1e-7))
            print('.',end = ' ')
        clear_output()
        showarray(img/255.0)

    ### Now returning Images ###
    return PIL.Image.fromarray(np.uint8(np.clip(img/255.0, 0, 1)*255))



image_name = 'path_to_image.jpg'

for l, layer in enumerate(layers):
    # Get layers and channels
    layer = layer.split("/")[1]
    num_channels = T(layer).shape[3]
    new_file_path = './'+layer+'/'

    # Make the directory for the layer
    if not os.path.exists(new_file_path):
        os.makedirs(new_file_path)

    # Go through every channel, deep dream the image, and return back to be saved.
    for i in range(num_channels): # might have an off by 1 error
        img0 = PIL.Image.open(image_name)
        img0 = np.float32(img0)
        deep_dream_image = render_deepdream(T(layer)[:,:,:,i], img0=img0)
        new_name = layer+'_channel_{num:0{width}}'.format(num=i, width=5)+'.jpg'
        deep_dream_image.save(new_file_path+new_name)

