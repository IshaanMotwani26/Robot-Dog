# Robot Dog Simulation

This project showcases the creation of a simple quadruped robot — a robot dog — built and simulated using [MuJoCo](https://mujoco.org/) and Python. The project demonstrates how to define a robot’s structure using MJCF XML and control it using a PD (Proportional-Derivative) controller for walking.

## Overview

The robot consists of:
- A central torso
- Four legs, each controlled by a single hinge joint and motor actuator
- A basic ground plane for contact and balance

The front and back legs alternate in phase, producing a simple walking gait in simulation.

## Core Components

dog.xml: Defines the robot’s body, legs, joints, and actuators using the MJCF XML format.

simulate_dog.py: Loads the model, sets up the simulation, and applies PD control to move the legs in a coordinated pattern.

## Technical Details

Physics Engine: MuJoCo  
Programming Language: Python  
Controller Type: Proportional Derivative  
Simulation Loop: Real-time viewer using `mujoco.viewer.launch_passive`  

## Features

- Real-time visualization of robot movement  
- Modular design — easily extendable to multi-joint legs  
- Tunable controller gains (Kp, Kd) for smoother or stiffer motion  
- Demonstrates floating-base robot simulation with MuJoCo  

## Author

Developed by **Ishaan Motwani**  
