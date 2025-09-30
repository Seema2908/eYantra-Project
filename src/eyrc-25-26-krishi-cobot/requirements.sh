#!/bin/bash
# Function to install packages
install_package() {
    sudo apt-get install -y "$1" > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo -e "\e[32mPackage: $1 installed successfully!\e[0m"
    else
        echo -e "\e[31mPackage: $1 failed to install:(\e[0m"
    fi
}

# Install necessary packages
echo -e "\e[34mInstalling required packages...\e[0m"
install_package "python3-pip"
install_package "ros-humble-ros-gz"
install_package "ros-humble-ros-gz-bridge"
install_package "ros-humble-ign-ros2-control"
install_package "ros-humble-teleop-twist-keyboard"
install_package "ros-humble-robot-state-publisher"
install_package "ros-humble-joint-state-publisher"
install_package "ros-humble-joint-state-publisher-gui"
install_package "ros-humble-ros2-controllers"
install_package "ros-humble-topic-tools"
install_package "ros-humble-xacro"
install_package "ros-humble-tf-transformations"
install_package "ros-humble-joint-state-broadcaster"
install_package "ros-humble-joint-trajectory-controller"
install_package "ros-humble-controller-manager"
install_package "ros-humble-gazebo-msgs"
install_package "ros-dev-tools"


pip3 install python-fcl > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo -e "\e[32mPackage: python-fcl installed successfully.\e[0m"
else
    echo -e "\e[33mWarning: Failed to install python-fcl.\e[0m"
fi

pip3 install urdf-parser-py > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo -e "\e[32mPackage: urdf-parser-py installed successfully.\e[0m"
else
    echo -e "\e[33mWarning: Failed to install urdf-parser-py.\e[0m"
fi

pip3 install networkx==3.4.2 > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo -e "\e[32mPackage: networkx installed successfully.\e[0m"
else
    echo -e "\e[33mWarning: Failed to install networkx.\e[0m"
fi

pip3 install transforms3d > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo -e "\e[32mPackage: transforms3d installed successfully.\e[0m"
else
    echo -e "\e[33mWarning: Failed to install transforms3d.\e[0m"
fi

pip3 install opencv-contrib-python==4.7.0.72 > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo -e "\e[32mPackage: opencv-contrib-python installed successfully.\e[0m"
else
    echo -e "\e[33mWarning: Failed to install opencv-contrib-python.\e[0m"
fi

pip3 install numpy==1.21.5 > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo -e "\e[32mPackage: numpy installed successfully.\e[0m"
else
    echo -e "\e[33mWarning: Failed to install numpy.\e[0m"
fi

pip3  install trimesh > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo -e "\e[32mPackage: trimesh installed successfully.\e[0m"
else
    echo -e "\e[33mWarning: Failed to install trimesh.\e[0m"
fi


echo -e "\e[32m---------------------------------\nSetup complete!\n---------------------------------\n\e[0m"
