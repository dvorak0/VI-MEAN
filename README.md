# VI-MEAN
Visual-Inertia-fusion-based Monocular dEnse mAppiNg

This repository corresponds to our paper on ICRA 2017:

__Real-time Monocular Dense Mapping on Aerial Robots using Visual-Inertial Fusion__

*[Zhenfei Yang](mailto:zyangag@connect.ust.hk), [Fei Gao](mailto:fgaoaa@connect.ust.hk), and [Shaojie Shen](mailto:eeshaojie@ust.hk)*

[paper](https://github.com/dvorak0/VI-MEAN/raw/master/ICRA17_1095_MS.pdf) and [video](https://www.youtube.com/watch?v=M4BMks6bQbc)

## How to view the 3D model
The following rosbags might be helpful to take a close look at the dense 3D model produced by our system.

https://www.dropbox.com/s/pwbw0cz1i26sjba/2016-09-18-15-46-27-part1.bag?dl=0

https://www.dropbox.com/s/5uboefps9js13bs/2016-09-18-16-07-25-part2.bag?dl=0

Topic of Meshed dense map: /Chisel/full_mesh

Topic of estimated path: /self_calibration_estimator/path

Note: Unfortunately, the model is too dense to be recorded real-time using our onboard computers. ROS randomly throw data with limited IO, resulting very low frequency of the messages in the bags.

We emphasize the map received by controller is updated at 10Hz. The frequency of map generation really matters a lot for autonomous systems.

## How to compile
1. Choose correct `CUDA_NVCC_FLAGS` in `stereo_mapper/CMakeLists.txt`
2. Install the following dependencies:
* OpenCV
* Eigen
* Ceres (http://ceres-solver.org/)
* Modified [OpenChisel](https://github.com/personalrobotics/OpenChisel) (already included in this repo), which requires PCL compiled with c++11. Please follow OpenChisel's instruction.
* Subset of [camodocal](https://github.com/hengli/camodocal) (already inclued in this repo with name of `camera_model`) to support wide-angle lens.
3. `catkin_make`
## How to run it with our data
1. Download sample.bag via: http://uav.ust.hk/storage/sample.bag
2. `roslaunch stereo_mapper sample_all.launch`
3. `rosbag play path_to_the_bag/sample.bag`
4. Use rviz with your config file `sample.rviz` to visualize
## How to run it with your data
1. Calibrate your camera using `camera_model` to get the calibration file that looks like `sample_camera_calib.yaml`
2. Modify the launch file to fit your topic names and the path to your calibration file
3. Follow the above steps

## Licence
VI-MEAN is licensed under the GNU General Public License Version 3 (GPLv3), see http://www.gnu.org/licenses/gpl.html.

For commercial purposes, a more efficient version under different licencing terms is under development.

