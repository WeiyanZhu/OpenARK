import sys
import numpy as np
import pyzed.sl as sl
import cv2
from pathlib import Path
import os
from PIL import Image

def progress_bar(percent_done, bar_length=50):
    done_length = int(bar_length * percent_done / 100)
    bar = '=' * done_length + '-' * (bar_length - done_length)
    sys.stdout.write('[%s] %f%s\r' % (bar, percent_done, '%'))
    sys.stdout.flush()


def main():

    if len(sys.argv) < 2:
        print("Please specify path to .svo file.")
        exit()

    filepath = sys.argv[1]
    print("Starting...")
    #initialize camera
    input_type = sl.InputType()
    input_type.set_from_svo_file(filepath)
    init = sl.InitParameters(input_t=input_type, svo_real_time_mode=False, depth_mode= sl.DEPTH_MODE.ULTRA, coordinate_units=sl.UNIT.METER)#, coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP)
    cam = sl.Camera()
    status = cam.open(init)

    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    tracking_parameters = sl.PositionalTrackingParameters()
    err = cam.enable_positional_tracking(tracking_parameters)
    #set paths to save data
    path_prefix = Path(os.path.dirname(__file__))/"svoData"
    depth_image_folder = path_prefix/ "depth"
    rgb_image_folder = path_prefix/ "RGBpng"
    rgb_JPG_folder = path_prefix/ "RGB"
    pose_folder = path_prefix/ "tcw"

    path_prefix.mkdir(parents=True, exist_ok=True)
    depth_image_folder.mkdir(parents=True, exist_ok=True)
    rgb_image_folder.mkdir(parents=True, exist_ok=True)
    pose_folder.mkdir(parents=True, exist_ok=True)
    rgb_JPG_folder.mkdir(parents=True, exist_ok=True)

    print("Processing SVO file: {0}".format(filepath))
    # Prepare single image containers
    depth_image = sl.Mat()
    rgb_image = sl.Mat()
    #go through svo files to get info
    runtime_param = sl.RuntimeParameters()
    runtime_param.sensing_mode = sl.SENSING_MODE.STANDARD

    total_frames = cam.get_svo_number_of_frames()
    '''print out extrinsic values'''
    ImuToLeft = cam.get_camera_information(resizer=sl.Resolution(640,480)).camera_imu_transform
    leftToRight = cam.get_camera_information(resizer=sl.Resolution(640,480)).calibration_parameters.stereo_transform
    trans = np.matmul(ImuToLeft.m, leftToRight.m)
    trans = np.linalg.inv(trans)
    print("right to imu pose", trans)
    '''
    lc = cam.get_camera_information(resizer=sl.Resolution(640,480)).calibration_parameters.left_cam
    print("intrinsic left", [lc.fx, lc.fy, lc.cx, lc.cy])
    trans = cam.get_camera_information(resizer=sl.Resolution(640,480)).camera_imu_transform
    trans.inverse()
    print("imu to left camera trans", trans)'''

    while True:  
        err = cam.grab(runtime_param)
        
        if err == sl.ERROR_CODE.SUCCESS:
            frame = cam.get_svo_position()

            # Retrieve depth meastures
            cam.retrieve_measure(depth_image, sl.MEASURE.DEPTH)
            #cam.retrieve_image(depth_image, sl.VIEW.DEPTH) 
            cam.retrieve_image(rgb_image, sl.VIEW.LEFT)

            #store images
            depth_image_path = depth_image_folder / ("%s.png" % str(frame))#.zfill(5))
            rgb_image_path = rgb_image_folder / ("%s.png" % str(frame))#.zfill(5))
            resizedDepth = cv2.resize((depth_image.get_data()*1000).astype(np.uint16), (640, 480))#cv2.resize(depth_image.get_data().astype(np.uint16), (640, 480))
            resizedRGB = cv2.resize(rgb_image.get_data(), (640, 480))
            cv2.imwrite(str(depth_image_path), resizedDepth)
            cv2.imwrite(str(rgb_image_path), resizedRGB)

            #convert rgb to jpg
            dst = rgb_JPG_folder/(str(frame) + ".jpg")
            im1 = Image.open(rgb_image_folder / ("%s.png" % str(frame)))
            rgb_im = im1.convert('RGB')
            rgb_im.save(dst)

            #store pose
            '''
            sensors_data = sl.SensorsData()
            cam.get_sensors_data(sensors_data, time_reference = sl.TIME_REFERENCE.IMAGE)
            imu = sl.IMUData()
            imu = sensors_data.get_imu_data()

            zed_imu_pose = sl.Transform()
            imu.get_pose(zed_imu_pose)
            
            poseR = zed_imu_pose.get_rotation_matrix().r
            poseT = zed_imu_pose.get_translation().get()
            p = np.concatenate((poseR, poseT[:,None]), axis=1)
            p = np.concatenate((p, np.array([[0,0,0,1]])), axis=0)
            '''

            pose = sl.Pose()
            print(cam.get_position(pose, sl.REFERENCE_FRAME.WORLD))
            poseTransform = sl.Transform()
            pose.pose_data(poseTransform)
            poseRotation = poseTransform.get_rotation_matrix().r
            poseTranslation = poseTransform.get_translation().get()
            p = np.concatenate((poseRotation, poseTranslation[:,None]), axis=1)
            p = np.concatenate((p, np.array([[0,0,0,1]])), axis=0)
            print(p)
            #p = np.linalg.inv(p)

            pose_path = pose_folder / ("%s.txt" % str(frame))#.zfill(5))
            pose_file = open(pose_path, 'w')
            pose_file.write("%s %s %s %s\n" % (p[0][0], p[0][1], p[0][2], p[0][3]))#(p[0][3])/1000))
            pose_file.write("%s %s %s %s\n" % (p[1][0], p[1][1], p[1][2], p[1][3]))#float(p[1][3])/1000))
            pose_file.write("%s %s %s %s\n" % (p[2][0], p[2][1], p[2][2], p[2][3]))#float(p[2][3])/1000))
            pose_file.write("%s %s %s %s\n" % (p[3][0], p[3][1], p[3][2], p[3][3]))

        # Display progress
        progress_bar((frame + 1) / total_frames * 100, 30)

        # Check if we have reached the end of the video
        if frame >= (total_frames - 1):  # End of SVO
            sys.stdout.write("\nSVO end has been reached. Exiting now.\n")
            break

    cam.close()
    print("\nFINISH")



if __name__ == "__main__":
    main()