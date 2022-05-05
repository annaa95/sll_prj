## Usage

Once you have started a simulation adn set the robot controller to `<extern>`, you can use the following launch file to setup all the required ROS parameters and start the simulated SLL robot to ROS interface:

```
roslaunch sll_extern_roscontroller sll_ros.launch
```


## Multi Robots

If your simulation uses more than one robot controller, the `---node-name` controller arguments should be set to avoid a name clash between nodes.

