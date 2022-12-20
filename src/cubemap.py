import os
import sys
import time
import math
import zipfile
import imageio
import numpy as np
from PIL import Image
from numpy import clip
import matplotlib.pyplot as plt 
from math import pi,sin,cos,tan,atan2,hypot,floor

def return_files(path_input):

    # list to store files
    res = []

    # Iterate directory
    for path in os.listdir(path_input):
        # check if current path is a file
        if os.path.isfile(os.path.join(path_input, path)):
            res.append(path)
    return res

def generate_cubmaps(path_input, path_output):
    if not os.path.exists(path_output):
        os.makedirs(path_output) 

    # list to store files
    res = return_files(path_input)

    for i in range (0, len(res)):
        print('image: ', res[i])
        imgIn = Image.open(path_input + res[i])
        print(imgIn.size)
        inSize = imgIn.size
        imgOut = Image.new("RGB",(inSize[0], int(inSize[0]*3/4)),"black")
        convertBack(imgIn,imgOut)
        print(imgOut.size)
        imgOut.save(path_output + res[i])

# get x,y,z coords from out image pixels coords
# i,j are pixel coords
# face is face number
# edge is edge length
def outImgToXYZ(i,j,face,edge):
    a = 2.0*float(i)/edge
    b = 2.0*float(j)/edge
    if face==0: # back
        (x,y,z) = (-1.0, 1.0-a, 3.0 - b)
    elif face==1: # left
        (x,y,z) = (a-3.0, -1.0, 3.0 - b)
    elif face==2: # front
        (x,y,z) = (1.0, a - 5.0, 3.0 - b)
    elif face==3: # right
        (x,y,z) = (7.0-a, 1.0, 3.0 - b)
    elif face==4: # top
        (x,y,z) = (b-1.0, a -5.0, 1.0)
    elif face==5: # bottom
        (x,y,z) = (5.0-b, a-5.0, -1.0)
    return (x,y,z)

# convert using an inverse transformation
def convertBack(imgIn,imgOut):
    start = time.time()
    print("Starting...")
    inSize = imgIn.size
    outSize = imgOut.size
    inPix = imgIn.load()
    outPix = imgOut.load()
    edge = inSize[0]/4   # the length of each edge in pixels
    for i in range(outSize[0]):
        face = int(i/edge) # 0 - back, 1 - left 2 - front, 3 - right
        #print('face: ', face)
        if face==2:
            rng = range(0,int(edge*3))
        else:
            rng = range(int(edge), int(edge) * 2)

        for j in rng:
            if j<edge:
                face2 = 4 # top
            elif j>=2*edge:
                face2 = 5 # bottom
            else:
                face2 = face

            (x,y,z) = outImgToXYZ(i,j,face2,edge)
            theta = atan2(y,x) # range -pi to pi
            r = hypot(x,y)
            phi = atan2(z,r) # range -pi/2 to pi/2
            # source img coords
            uf = ( 2.0*edge*(theta + pi)/pi )
            vf = ( 2.0*edge * (pi/2 - phi)/pi)
            # Use bilinear interpolation between the four surrounding pixels
            ui = floor(uf)  # coord of pixel to bottom left
            vi = floor(vf)
            u2 = ui+1       # coords of pixel to top right
            v2 = vi+1
            mu = uf-ui      # fraction of way across pixel
            nu = vf-vi
            # Pixel values of four corners
            A = inPix[ui % inSize[0],int(clip(vi,0,inSize[1]-1))]
            B = inPix[u2 % inSize[0],int(clip(vi,0,inSize[1]-1))]
            C = inPix[ui % inSize[0],int(clip(v2,0,inSize[1]-1))]
            D = inPix[u2 % inSize[0],int(clip(v2,0,inSize[1]-1))]
            # interpolate
            (r,g,b) = (
              A[0]*(1-mu)*(1-nu) + B[0]*(mu)*(1-nu) + C[0]*(1-mu)*nu+D[0]*mu*nu,
              A[1]*(1-mu)*(1-nu) + B[1]*(mu)*(1-nu) + C[1]*(1-mu)*nu+D[1]*mu*nu,
              A[2]*(1-mu)*(1-nu) + B[2]*(mu)*(1-nu) + C[2]*(1-mu)*nu+D[2]*mu*nu )

            outPix[i,j] = (int(round(r)),int(round(g)),int(round(b)))
    end = time.time()
    print('elapsed time: ', end-start)
import pdb
def split_cubmaps(infile, outfile, keyword='cubemap'):
    if not os.path.exists(outfile):
        os.makedirs(outfile)
    '''
    if len(sys.argv) < 2:
        print("Usage: cubemap-cut.py <filename.jpg|png>")
        sys.exit(-1)
    '''
    #infile = root_path + '000009_cub.png'
    filename, original_extension = os.path.splitext(infile)
    file_extension = ".png"

    name_map = [ \
         ["", "", "posy", ""],
         ["negz", "negx", "posz", "posx"],
         ["", "", "negy", ""]]

    im_name = outfile.split('/')[-1]
    output_folder_path = outfile.split('/')[:-1]
    output_folder_path = '/'.join(output_folder_path) 
    try:
        im = Image.open(infile)
        print(infile, im.format, "%dx%d" % im.size, im.mode)

        width, height = im.size

        cube_size = width / 4

        filelist = []
        for row in range(3):
            for col in range(4):
                if name_map[row][col] != "":
                    sx = cube_size * col
                    sy = cube_size * row
                    fn = name_map[row][col] + file_extension
                    filelist.append(fn)
                    print("%s --> %s" % (str((sx, sy, sx + cube_size, sy + cube_size)), fn))
                    im.crop((sx, sy, sx + cube_size, sy + cube_size)).save(
                        output_folder_path + '/' + keyword + '_' + im_name + '_' + fn) 
    except IOError:
        pass

def split_cub_imgs(path_input, path_output):
    if not os.path.exists(path_output):
        os.makedirs(path_output)
    
    file_list = return_files(path_input)
    for i in range (0, len(file_list)):
        print(file_list[i])
        infile = file_list[i]
        split_cubmaps(path_input + infile, path_output + infile.split('.')[0])
        
def join_images(path_segmentation):
    # list to store files
    res = []

    # Iterate directory
    for path in os.listdir(path_segmentation):
        # check if current path is a file
        if os.path.isfile(os.path.join(path_segmentation, path)):
            res.append(path)
    print(res)

    vec_img = []
    for i in range(0, 6):
        #print(i)
        vec_img.append(Image.open(path_segmentation + res[i]))

    name_map = [ \
         ["", "posy", "", ""],
         ["negz", "negx", "posz", "posx"],
         ["", "negy", "", ""]]

    width, height = np.array(vec_img[0]).shape
    print(width, height)

    cube_size = np.zeros((width * 3 , width * 4))
    print(cube_size.shape)

    cube_size[0:width, width:width*2] = np.array(vec_img[4])
    cube_size[width:width*2, 0:width] = np.array(vec_img[0])
    cube_size[width:width*2, width:width*2] = np.array(vec_img[5])
    cube_size[width:width*2, width*2:width*3] = np.array(vec_img[3])
    cube_size[width:width*2, width*3:width*4] = np.array(vec_img[2])
    cube_size[2*width:width*3, width:width*2] = np.array(vec_img[1])
    return cube_size

def spherical_coordinates(i, j, w, h):
    """ Returns spherical coordinates of the pixel from the output image. """
    theta = 2*float(i)/float(w)-1
    phi = 2*float(j)/float(h)-1
    # phi = lat, theta = long
    return phi*(math.pi/2), theta*math.pi


def vector_coordinates(phi, theta):
    """ Returns 3D vector which points to the pixel location inside a sphere. """
    return (math.cos(phi) * math.cos(theta),  # X
            math.sin(phi),                    # Y
            math.cos(phi) * math.sin(theta))  # Z


# Assign identifiers to the faces of the cube
FACE_Z_POS = 1  # Left
FACE_Z_NEG = 2  # Right
FACE_Y_POS = 3  # Top
FACE_Y_NEG = 4  # Bottom
FACE_X_NEG = 5  # Front
FACE_X_POS = 6  # Back


def get_face(x, y, z):
    """ Uses 3D vector to find which cube face the pixel lies on. """
    largest_magnitude = max(abs(x), abs(y), abs(z))
    if largest_magnitude - abs(x) < 0.00001:
        return FACE_X_POS if x < 0 else FACE_X_NEG
    elif largest_magnitude - abs(y) < 0.00001:
        return FACE_Y_POS if y < 0 else FACE_Y_NEG
    elif largest_magnitude - abs(z) < 0.00001:
        return FACE_Z_POS if z < 0 else FACE_Z_NEG


def raw_face_coordinates(face, x, y, z):
    """
    Return coordinates with necessary sign (- or +) depending on which face they lie on.
    From Open-GL specification (chapter 3.8.10) https://www.opengl.org/registry/doc/glspec41.core.20100725.pdf
    """
    if face == FACE_X_NEG:
        xc = z
        yc = y
        ma = x
        return xc, yc, ma
    elif face == FACE_X_POS:
        xc = -z
        yc = y
        ma = x
        return xc, yc, ma
    elif face == FACE_Y_NEG:
        xc = z
        yc = -x
        ma = y
        return xc, yc, ma
    elif face == FACE_Y_POS:
        xc = z
        yc = x
        ma = y
        return xc, yc, ma
    elif face == FACE_Z_POS:
        xc = x
        yc = y
        ma = z
        return xc, yc, ma
    elif face == FACE_Z_NEG:
        xc = -x
        yc = y
        ma = z
        return xc, yc, ma


def raw_coordinates(xc, yc, ma):
    """ Return 2D coordinates on the specified face relative to the bottom-left corner of the face. Also from Open-GL spec."""
    return (float(xc)/abs(float(ma)) + 1) / 2, (float(yc)/abs(float(ma)) + 1) / 2


def face_origin_coordinates(face, n):
    """ Return bottom-left coordinate of specified face in the input image. """
    if face == FACE_X_NEG:
        return n, n
    elif face == FACE_X_POS:
        return 3*n, n
    elif face == FACE_Z_NEG:
        return 2*n, n
    elif face == FACE_Z_POS:
        return 0, n
    elif face == FACE_Y_POS:
        return n, 0
    elif face == FACE_Y_NEG:
        return n, 2*n


def normalized_coordinates(face, x, y, n):
    """ Return pixel coordinates in the input image where the specified pixel lies. """
    face_coords = face_origin_coordinates(face, n)
    normalized_x = math.floor(x*n)
    normalized_y = math.floor(y*n)

    # Stop out of bound behaviour
    if normalized_x < 0:
        normalized_x = 0
    elif normalized_x >= n:
        normalized_x = n-1
    if normalized_y < 0:
        normalized_x = 0
    elif normalized_y >= n:
        normalized_y = n-1

    return face_coords[0] + normalized_x, face_coords[1] + normalized_y


def find_corresponding_pixel(i, j, w, h, n):
    """
    :param i: X coordinate of output image pixel
    :param j: Y coordinate of output image pixel
    :param w: Width of output image
    :param h: Height of output image
    :param n: Height/Width of each square face
    :return: Pixel coordinates for the input image that a specified pixel in the output image maps to.
    """

    spherical = spherical_coordinates(i, j, w, h)
    vector_coords = vector_coordinates(spherical[0], spherical[1])
    face = get_face(vector_coords[0], vector_coords[1], vector_coords[2])
    raw_face_coords = raw_face_coordinates(face, vector_coords[0], vector_coords[1], vector_coords[2])

    cube_coords = raw_coordinates(raw_face_coords[0], raw_face_coords[1], raw_face_coords[2])

    return normalized_coordinates(face, cube_coords[0], cube_coords[1], n)

def convert_img(infile, outfile):
    inimg = Image.open(infile)

    wo, ho = inimg.size
    print('image size: ', wo, ho)

    # Calculate height and width of output image, and size of each square face
    #h = int(wo/3)
    #w = int(2*h)
    h = 2688
    w = 5376
    n = ho/3
    print(w,h)


    # Create new image with width w, and height h
    outimg = Image.new('RGB', (w, h))

    # For each pixel in output image find colour value from input image
    for ycoord in range(0, h):
        for xcoord in range(0, w):
            corrx, corry = find_corresponding_pixel(xcoord, ycoord, w, h, n)

            outimg.putpixel((xcoord, ycoord), inimg.getpixel((corrx, corry)))
        # Print progress percentage
        #print(str(round((float(ycoord)/float(h))*100, 2)) + '%')


    outimg.save(outfile, 'PNG')