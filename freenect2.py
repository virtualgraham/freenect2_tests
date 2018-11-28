# coding: utf-8

# An example using startStreams
import time
import numpy as np
import cv2
import sys
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame
from numpy import inf
import skimage.measure
import pathlib

try:
    from pylibfreenect2 import OpenGLPacketPipeline
    pipeline = OpenGLPacketPipeline()
except:
    try:
        from pylibfreenect2 import OpenCLPacketPipeline
        pipeline = OpenCLPacketPipeline()
    except:
        from pylibfreenect2 import CpuPacketPipeline
        pipeline = CpuPacketPipeline()

print("Packet pipeline:", type(pipeline).__name__)


pathlib.Path('/Users/user/Desktop/sandbox/imgs/color/').mkdir(parents=True, exist_ok=True) 
pathlib.Path('/Users/user/Desktop/sandbox/imgs/normal/').mkdir(parents=True, exist_ok=True) 
pathlib.Path('/Users/user/Desktop/sandbox/imgs/depth/').mkdir(parents=True, exist_ok=True) 

enable_rgb = True
enable_depth = True

fn = Freenect2()
num_devices = fn.enumerateDevices()
if num_devices == 0:
    print("No device connected!")
    sys.exit(1)

serial = fn.getDeviceSerialNumber(0)
device = fn.openDevice(serial, pipeline=pipeline)

types = 0
if enable_rgb:
    types |= FrameType.Color
if enable_depth:
    types |= (FrameType.Ir | FrameType.Depth)
listener = SyncMultiFrameListener(types)

# Register listeners
device.setColorFrameListener(listener)
device.setIrAndDepthFrameListener(listener)

if enable_rgb and enable_depth:
    device.start()
else:
    device.startStreams(rgb=enable_rgb, depth=enable_depth)

# NOTE: must be called after device.start()
if enable_depth:
    registration = Registration(device.getIrCameraParams(),
                                device.getColorCameraParams())


# 1,2,3,4,5,6,8,9,10,12,15,18,20,24,27,30
reduction_factor = 4 


out_height = 1080//reduction_factor # max 424

frames_per_second = 3 # max 24



out_width = round(512 * (out_height/424))

resize_width = round(1920. * (out_height/1080.))
crop_left = (resize_width-out_width)//2
crop_right = out_width + crop_left

print(out_width, out_height)
print(resize_width, out_height)
print(crop_left, crop_right)

undistorted = Frame(512, 424, 4)
registered = Frame(512, 424, 4)
big_depth = Frame(1920, 1082, 4)

# full_color_depth_map = np.full((424, 512), -1, np.int32)
# color_depth_map = np.zeros((424, 512), np.int32)

last_frame_time = time.time() * 1000

registered_fill = np.zeros((424, 512, 3), np.uint8)

def blah(a, axis=None):
    print(a, axis)
    return np.min(a, axis)

def get_normals(d_im):

    print(d_im.shape)

    d_im = d_im.astype("float64")

    zy, zx = np.gradient(d_im)  

    zx = cv2.Sobel(d_im, cv2.CV_64F, 1, 0, ksize=7)     
    zy = cv2.Sobel(d_im, cv2.CV_64F, 0, 1, ksize=7)

    normal = np.dstack((-zx, -zy, np.ones_like(d_im)))
    n = np.linalg.norm(normal, axis=2)
    normal[:, :, 0] /= n
    normal[:, :, 1] /= n
    normal[:, :, 2] /= n

    # offset and rescale values to be in 0-255
    normal += 1
    normal /= 2
    normal *= 255

    return normal[:, :, ::-1].astype(np.uint8)

while True:
    frames = listener.waitForNewFrame()
    current_time = time.time() * 1000

    if (current_time - last_frame_time) >= 1000./frames_per_second:
        color = frames["color"]
        ir = frames["ir"]
        depth = frames["depth"]

        registration.apply(color, depth, undistorted, registered, bigdepth=big_depth)
       
        #################################
        # Old Stuff
        #################################

        #registration.apply(color, depth, undistorted, registered, color_depth_map=color_depth_map.ravel())
        
        # c = color.asarray()
        # r = registered.asarray(np.uint8)


        # cv2.imshow("registered", r)

        # cv2.imshow("undistorted", u)

        # mean_sum = 0.
        # mean_count = 0.

        # for index, x in np.ndenumerate(full_color_depth_map):
            
        #     y = color_depth_map[index]

        #     if y != -1:
        #         full_color_depth_map[index] = y
        #         registered_fill[index] = c[np.unravel_index(y, (1080, 1920))][0:3]
        #     elif x != -1:
        #         registered_fill[index] = c[np.unravel_index(x, (1080, 1920))][0:3]
        #     else:
        #         registered_fill[index] = [0,0,0]

        # cv2.imshow("registered_fill", registered_fill )
    
        # u = undistorted.asarray(np.float32)
        # u = (u/4500) * 2**16
        # u = u.astype(np.uint16)

        # uf = '/Users/user/Desktop/sandbox/imgs/undistorted/{0:.0f}.png'.format(current_time)
        
        # cv2.imwrite(uf, u) 

        # b = skimage.measure.block_reduce(b, (reduction_factor//2,reduction_factor//2), np.mean)
        # b[b == inf] = np.nan
        # b = cv2.resize(b, (resize_width, out_height), interpolation=cv2.INTER_NEAREST)
        # b_mini = skimage.measure.block_reduce(b, (reduction_factor*2,reduction_factor*2), np.mean)
        # b_mini = cv2.resize(b, (resize_width//2, out_height//2), interpolation=cv2.INTER_NEAREST)
        # cv2.imshow("b_mini", b_mini)
        # b_mini = cv2.inpaint(b_mini, np.array(b_mini == np.inf, dtype=np.uint8), 2, cv2.INPAINT_NS)
        #cv2.imshow("b_mini2", b_mini)
        #b_mini = cv2.resize(b_mini, (b.shape[1], b.shape[0]))
        #np.copyto(b, b_mini, where = (b == np.inf))
        #b[b == np.inf] = 0


        #################################
        # Resize, crop and save color image
        #################################
        c = color.asarray()[1:1081,:]
        c = cv2.resize(c, (resize_width, out_height), interpolation=cv2.INTER_NEAREST)
        c = c[:, crop_left:crop_right]
        cv2.imshow("color", c )

        cf = '/Users/user/Desktop/sandbox/imgs/color/{0:.0f}.png'.format(current_time)
        cv2.imwrite(cf, c[:,:,0:3]) 

        #################################
        # Depth Map
        #################################

        b = big_depth.asarray(np.float32)[1:1081,:]
        b = skimage.measure.block_reduce(b, (reduction_factor,reduction_factor), np.mean)
        b = b[:, crop_left:crop_right]
        b = (b/4500) * 2**16
        
        # INPAINT WITH REDUCE
        #################################       

        # b_mini = cv2.resize(b, (out_width//2, out_height//2), interpolation=cv2.INTER_NEAREST)
        # b_mini_mask = np.array(b_mini == np.inf, dtype=np.uint8)
        # b_mini = b_mini.astype(np.uint16)
        
        # b_mini = cv2.inpaint(b_mini, b_mini_mask, 5, cv2.INPAINT_NS)

        # cv2.imshow("b_mini", b_mini)

        # b_mini = cv2.resize(b_mini, (out_width, out_height))
    
        # np.copyto(b, b_mini, where = (b == np.inf))
        # b = b.astype(np.uint16)

        # INPAINT WITHOUT REDUCE
        ################################# 

        b_mask = np.array(b == np.inf, dtype=np.uint8)
        b = b.astype(np.uint16)
        b = cv2.inpaint(b, b_mask, 5, cv2.INPAINT_NS)
        

        # Finish Depth Map
        ################################# 

        cv2.imshow("big_depth", b)

        bf = '/Users/user/Desktop/sandbox/imgs/depth/{0:.0f}.png'.format(current_time)        
        cv2.imwrite(bf, np.reshape(b, (out_height, out_width, 1))) 

        #################################
        # Depth Map
        #################################

        n = get_normals(b)
        cv2.imshow('normals', n)

        nf = '/Users/user/Desktop/sandbox/imgs/normal/{0:.0f}.png'.format(current_time)
        cv2.imwrite(nf, n) 

        last_frame_time = current_time
    
        

    listener.release(frames)

    key = cv2.waitKey(delay=1)
    if key == ord('q'):
        break

device.stop()
device.close()

sys.exit(0)