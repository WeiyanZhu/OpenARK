import sys
import numpy as np
import pyzed.sl as sl
import cv2
from pathlib import Path
import os

def progress_bar(percent_done, bar_length=50):
    done_length = int(bar_length * percent_done / 100)
    bar = '=' * done_length + '-' * (bar_length - done_length)
    sys.stdout.write('[%s] %f%s\r' % (bar, percent_done, '%'))
    sys.stdout.flush()


def main():

    if len(sys.argv) != 3:
        print("Please specify path to .svo file, and imu file as paramters.")
        exit()

    filepath = sys.argv[1]
    imuFilepath = sys.argv[2]
    print("Starting...")
    #initialize camera
    input_type = sl.InputType()
    input_type.set_from_svo_file(filepath)
    init = sl.InitParameters(input_t=input_type, svo_real_time_mode=False)
    cam = sl.Camera()
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    #set paths to save data
    path_prefix = Path(os.path.dirname(__file__))/"svoData"
    left_image_folder = path_prefix/ "infrared"
    right_image_folder = path_prefix/ "infrared2"
    depth_image_folder = path_prefix/ "depth"
    rgb_image_folder = path_prefix/ "rgb"
    path_prefix.mkdir(parents=True, exist_ok=True)
    left_image_folder.mkdir(parents=True, exist_ok=True)
    right_image_folder.mkdir(parents=True, exist_ok=True)
    depth_image_folder.mkdir(parents=True, exist_ok=True)
    rgb_image_folder.mkdir(parents=True, exist_ok=True)
    timestamp_file = open(str(path_prefix/ "timestamp.txt"), 'w')

    #meta file's content is always the same
    meta_file = open(str(path_prefix/ "meta.txt"), 'w')
    meta_file.write("depth 0.001")
    meta_file.close()
    #store camera intrinsic in intrin.bin
    #intrin_file = open(str(path_prefix/ "intrin.bin"), 'w')

    #write to imu data
    print("Processing IMU file: {0}".format(imuFilepath))
    imu_file = open(str(path_prefix/ "imu.txt"), 'w')
    imu_data_file = open(imuFilepath, 'r')
    lineIndex = 0
    for line in imu_data_file:
        if(lineIndex == 0):
            imu_file.write("ts %s\n" % line.strip()[4:])
        elif(lineIndex == 1 or lineIndex == 2):
            typeText = "gy" if lineIndex == 1 else "ac"
            values = line.strip()[1:-1].split(',')
            imu_file.write("%s %s %s %s\n" % (typeText, values[0], values[1], values[2]))
        lineIndex = (lineIndex+1)%3
    imu_data_file.close()
    imu_file.close()


    print("Processing SVO file: {0}".format(filepath))
    # Prepare single image containers
    left_image = sl.Mat()
    right_image = sl.Mat()
    depth_image = sl.Mat()
    rgb_image = sl.Mat()
    #go through svo files to get info
    runtime_param = sl.RuntimeParameters()
    runtime_param.sensing_mode = sl.SENSING_MODE.FILL

    total_frames = cam.get_svo_number_of_frames()
    while True:  
        err = cam.grab(runtime_param)
        
        if err == sl.ERROR_CODE.SUCCESS:

            #store metadata
            frame = cam.get_svo_position()
            timestamp = cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
            timestamp_file.write("%s %s\n" % (frame, str(timestamp)[4:]))

            sensors_data = sl.SensorsData()
            cam.get_sensors_data(sensors_data, sl.TIME_REFERENCE.IMAGE)
            imu = sl.IMUData()
            imu = sensors_data.get_imu_data()
            sensor_timestamp = imu.timestamp.get_nanoseconds()
            ''' old way to get imu
            sensor_gyroscope = imu.get_angular_velocity()
            sensor_accelerometer = imu.get_linear_acceleration()

            imu_file.write("ts %s\n" % sensor_timestamp)
            imu_file.write("gy %s %s %s\n" % (sensor_gyroscope[0], sensor_gyroscope[1], sensor_gyroscope[2]))
            imu_file.write("ac %s %s %s\n" % (sensor_accelerometer[0], sensor_accelerometer[1], sensor_accelerometer[2]))
            '''
            # Retrieve SVO images
            cam.retrieve_image(left_image, sl.VIEW.LEFT_GRAY)
            cam.retrieve_image(right_image, sl.VIEW.RIGHT_GRAY)
            cam.retrieve_measure(depth_image, sl.MEASURE.DEPTH) #sl.MEASURE.DEPTH for 16bit depth map, need to ask what we need later
            #cam.retrieve_image(depth_image, sl.VIEW.DEPTH) 
            cam.retrieve_image(rgb_image, sl.VIEW.LEFT)

            #store images
            left_image_path = left_image_folder / ("%s.png" % str(frame).zfill(5))
            right_image_path = right_image_folder / ("%s.png" % str(frame).zfill(5))
            depth_image_path = depth_image_folder / ("%s.png" % str(frame).zfill(5))
            rgb_image_path = rgb_image_folder / ("%s.png" % str(frame).zfill(5))
            #resize images to (640,480) cause currently OpenARK works with this resolution
            resizedLeft = cv2.resize(left_image.get_data(), (640, 480))
            resizedRight = cv2.resize(right_image.get_data(), (640, 480))
            resizedDepth = cv2.resize(depth_image.get_data().astype(np.uint16), (640, 480))
            resizedRGB = cv2.resize(rgb_image.get_data(), (640, 480))
            cv2.imwrite(str(left_image_path), resizedLeft)
            cv2.imwrite(str(right_image_path), resizedRight)
            cv2.imwrite(str(depth_image_path), resizedDepth)
            cv2.imwrite(str(rgb_image_path), resizedRGB)

        # Display progress
        progress_bar((frame + 1) / total_frames * 100, 30)

        # Check if we have reached the end of the video
        if frame >= (total_frames - 1):  # End of SVO
            sys.stdout.write("\nSVO end has been reached. Exiting now.\n")
            break

    cam.close()
    timestamp_file.close()
    imu_file.close()
    print("\nFINISH")



if __name__ == "__main__":
    main()