# VI-MEAN
Visual-Inertia-fusion-based Monocular dEnse mAppiNg

This repository corresponds to our ICRA 2017 submission:

__Real-time Monocular Dense Mapping on Aerial Robots using Visual-Inertial Fusion__
*Zhenfei Yang, Fei Gao, and Shaojie Shen*

We are still working on making the code easy to read, compile, and run.

The following rosbags might be helpful to take a close look at the dense 3D model produced by our system.

https://www.dropbox.com/s/pwbw0cz1i26sjba/2016-09-18-15-46-27-part1.bag?dl=0

https://www.dropbox.com/s/5uboefps9js13bs/2016-09-18-16-07-25-part2.bag?dl=0

Topic of Meshed dense map: /Chisel/full_mesh

Topic of estimated path: /self_calibration_estimator/path

Note: Unfortunately, the model is too dense to be recorded real-time using our onboard computers. ROS randomly throw data when the IO is limited, resulting very low frequency of the messages in the bags.

We emphasize the map received by controller is updated at 10Hz. The frequency of map generation really matters a lot for autonomous systems.



