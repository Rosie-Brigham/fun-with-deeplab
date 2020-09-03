import json
import urllib.request
from PIL import Image
from PIL import ImageOps


# This gets all the images from the file - saves them, and flips them
with open('new-images.json') as data_file:    
    data = json.load(data_file)
    i = 0
    for image in data:
        i += 1
        print(image['Label'])
        print("")
        # If there is an image..
        if len(image['Label']) > 0:
            
            # Save the image
            imageURL = image['Labeled Data']
            # import code; code.interact(local=dict(globals(), **locals()))

            img = urllib.request.urlretrieve(imageURL, f"deeplab/datasets/PQR/JPEGImages/{i}.jpeg")
            img.convert("RGB").save(f"deeplab/datasets/PQR/JPEGImages/{i}.jpeg")
            
            # then flip it...
            im = Image.open(f"deeplab/datasets/PQR/JPEGImages/{i}.jpeg")            
            im = ImageOps.mirror(im)
            im = im.convert("RGB")
            im.save(f"deeplab/datasets/PQR/JPEGImages/{i}m.jpeg")
            im = ImageOps.flip(im)
            im = im.convert("RGB")
            im.save(f"deeplab/datasets/PQR/JPEGImages/{i}f.jpeg")
            im = ImageOps.mirror(im)
            im = im.convert("RGB")
            im.save(f"deeplab/datasets/PQR/JPEGImages/{i}fm.jpeg")

            # # then save the label
            segmented = image['Label']['objects'][0]['instanceURI']
            # import code; code.interact(local=dict(globals(), **locals()))

            img = urllib.request.urlretrieve(segmented, f"deeplab/datasets/PQR/SegmentationClass/{i}.png")
            img.convert("RGB").save(f"deeplab/datasets/PQR/SegmentationClass/{i}.png")
            # then flip it...
            im = Image.open(f"deeplab/datasets/PQR/SegmentationClass/{i}.png")
            im = ImageOps.mirror(im)
            im.save(f"deeplab/datasets/PQR/SegmentationClass/{i}m.png")
            im = ImageOps.flip(im)
            im.save(f"deeplab/datasets/PQR/SegmentationClass/{i}f.png")
            im = ImageOps.mirror(im)
            im.save(f"deeplab/datasets/PQR/SegmentationClass/{i}fm.png")

            # import code; code.interact(local=dict(globals(), **locals()))
