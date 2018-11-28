# coding: utf-8

# An example using startStreams
import time
import numpy as np
import cv2
import sys
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame
from numpy import inf
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

frames_per_second = 3 # max 24




undistorted = Frame(512, 424, 4)
registered = Frame(512, 424, 4)
big_depth = Frame(1920, 1082, 4)



last_frame_time = time.time() * 1000



while True:

    frames = listener.waitForNewFrame()
    current_time = time.time() * 1000

    if (current_time - last_frame_time) >= 1000./frames_per_second:
        
        color = frames["color"]
        depth = frames["depth"]

        registration.apply(color, depth, undistorted, registered, bigdepth=big_depth)
       
        #################################
        # Color
        #################################

        cf = '/Users/user/sandbox/imgs/color/{0:.0f}.png'.format(current_time)
        cv2.imwrite(cf, c[:,:,0:3]) 

        #################################
        # Depth
        #################################

        b = big_depth.asarray(np.float32)[1:1081,:]
        b = (b/4500) * 2**16
        b = b.astype(np.uint16)

        bf = '/Users/user/sandbox/imgs/depth/{0:.0f}.png'.format(current_time)        
        cv2.imwrite(bf, image[..., np.newaxis]) 


        last_frame_time = current_time
    
        
    listener.release(frames)

    key = cv2.waitKey(delay=1)
    if key == ord('q'):
        break

device.stop()
device.close()

sys.exit(0)