import math
import os
import sys
import time

import numpy as np
import pybullet as p
import pybullet_data
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

# Add the path to import robot classes from pybullet_ur5_robotiq
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "pybullet_ur5_robotiq"))
from robot import UR5Robotiq85
from utilities import Camera


class CanGraspingEnv:
    """Environment for can grasping task using UR5 robot"""

    def __init__(self, vis=True):
        self.vis = vis
        # Connect to physics server
        self.physics_client = p.connect(p.GUI if vis else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Set up physics
        p.setGravity(0, 0, -9.8)
        self.plane_id = p.loadURDF("plane.urdf")

        # Set up camera
        self.initial_cam_pos = [-0.5, -0.285, 1.5]
        self.initial_cam_tar = [0.5, 0, 0.8]
        self.camera = Camera(
            cam_pos=self.initial_cam_pos,
            cam_tar=self.initial_cam_tar,
            cam_up_vector=[0, 0, 1],
            near=0.1,
            far=5.0,
            size=(224, 224),
            fov=60,
        )

        # Add camera position sliders
        self.cam_x_slider = p.addUserDebugParameter("cam_x", -3, 3, self.initial_cam_pos[0])
        self.cam_y_slider = p.addUserDebugParameter("cam_y", -3, 3, self.initial_cam_pos[1])
        self.cam_z_slider = p.addUserDebugParameter("cam_z", 0.5, 3, self.initial_cam_pos[2])
        self.cam_tar_x_slider = p.addUserDebugParameter("cam_target_x", -2, 2, self.initial_cam_tar[0])
        self.cam_tar_y_slider = p.addUserDebugParameter("cam_target_y", -2, 2, self.initial_cam_tar[1])
        self.cam_tar_z_slider = p.addUserDebugParameter("cam_target_z", 0, 2, self.initial_cam_tar[2])

        # Initialize robot
        robot_pos = [0, 0, 0.62]  # Position robot on top of table (table height ~0.62m)
        robot_ori = [0, 0, 1.57]
        self.robot = UR5Robotiq85(robot_pos, robot_ori)
        self.robot.load()
        self.robot.step_simulation = self.step_simulation

        # Load table and can
        self.setup_scene()

    def setup_scene(self):
        """Set up the table and can in the scene"""
        # Load table
        table_pos = [0.5, 0, 0]
        self.table_id = p.loadURDF("table/table.urdf", table_pos, useFixedBase=True)

        # Load can
        can_pos = [0.68, 0, 0.65]  # On top of the table
        self.can_id = p.loadURDF("can.urdf", can_pos)
        self.x_slider = p.addUserDebugParameter("can_x", -1, 1, can_pos[0])
        self.y_slider = p.addUserDebugParameter("can_y", -1, 1, can_pos[1])
        self.z_slider = p.addUserDebugParameter("can_z", 0, 2, can_pos[2])

    def update_camera_position(self):
        """Update camera position based on slider values"""
        cam_x = p.readUserDebugParameter(self.cam_x_slider)
        cam_y = p.readUserDebugParameter(self.cam_y_slider)
        cam_z = p.readUserDebugParameter(self.cam_z_slider)
        cam_tar_x = p.readUserDebugParameter(self.cam_tar_x_slider)
        cam_tar_y = p.readUserDebugParameter(self.cam_tar_y_slider)
        cam_tar_z = p.readUserDebugParameter(self.cam_tar_z_slider)

        # Update camera with new position
        self.camera = Camera(
            cam_pos=[cam_x, cam_y, cam_z],
            cam_tar=[cam_tar_x, cam_tar_y, cam_tar_z],
            cam_up_vector=[0, 0, 1],
            near=0.1,
            far=5.0,
            size=(224, 224),
            fov=60,
        )

    def step_simulation(self):
        """Step the physics simulation"""
        p.stepSimulation()
        if self.vis:
            time.sleep(1.0 / 60.0)

    def get_camera_image(self):
        """Get RGB image from camera"""
        rgb, depth, seg = self.camera.shot()
        # Convert to numpy array and remove alpha channel if present
        rgb_array = np.array(rgb, dtype=np.uint8)
        if rgb_array.shape[-1] == 4:
            rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def get_current_ee_pose(self):
        """Get current end-effector pose [x, y, z, roll, pitch, yaw]"""
        ee_state = p.getLinkState(self.robot.id, self.robot.eef_id)
        ee_pos = ee_state[0]  # Position
        ee_orn_quat = ee_state[1]  # Orientation as quaternion
        ee_orn_euler = p.getEulerFromQuaternion(ee_orn_quat)  # Convert to euler angles
        return list(ee_pos) + list(ee_orn_euler)

    def apply_incremental_action(self, delta_action):
        """Apply incremental action to current end-effector pose"""
        current_pose = self.get_current_ee_pose()

        # Add the incremental action to current pose
        # Assuming delta_action is [dx, dy, dz, droll, dpitch, dyaw, gripper]
        new_pose = []
        for i in range(6):  # x, y, z, roll, pitch, yaw
            new_pose.append(current_pose[i] + delta_action[i])

        # Handle gripper action (usually the 7th element)
        gripper_action = delta_action[6] if len(delta_action) > 6 else 0

        return new_pose, gripper_action

    def reset(self):
        """Reset the environment"""
        self.robot.reset()
        # Reset can position
        p.resetBasePositionAndOrientation(self.can_id, [0.5, 0, 0.65], [0, 0, 0, 1])
        # Wait for physics to settle
        for _ in range(10):
            self.step_simulation()

    def close(self):
        """Close the environment"""
        p.disconnect(self.physics_client)


def get_action_from_openvla(image, prompt):
    """
    Placeholder for the OpenVLA model.
    Replace this with your actual model inference code.
    """
    print(f"VLA received prompt: '{prompt}'")

    # --- This is where your actual model logic would go ---
    # For this example, let's just return a fixed target action
    # based on a simple keyword in the prompt.
    if "move to can" in prompt or "grasp can" in prompt:
        # The can is at [0.5, 0, 0.65], let's target slightly above it
        # Return end-effector pose: [x, y, z, roll, pitch, yaw]
        target_action = [0.5, 0, 0.75, 0, math.pi / 2, 0]  # Above the can
    elif "pick up" in prompt:
        # Lower position to actually grasp
        target_action = [0.68, 0, 0.70, 0, math.pi / 2, 0]  # At can level
    else:
        # Default position
        target_action = [0.4, 0.2, 0.8, 0, math.pi / 2, 0]

    print(f"VLA model outputs target action: {target_action}")
    return target_action


def can_grasping_demo():
    """Main demo function following the workflow from main.py"""

    # Initialize environment
    env = CanGraspingEnv(vis=True)

    # Reset environment
    obs = env.reset()
    INSTRUCTION = "pick up the can"
    # Load Processor & VLA
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to("cuda:0")
    prompt = "In: What action should the robot take to {INSTRUCTION}?\nOut:"

    # Get command from user
    # prompt = input("Enter a command (e.g., 'move to can', 'grasp can', 'pick up can'): ")

    try:
        # Main control loop
        step_count = 0
        max_steps = 10000
        action_executed = False

        while step_count < max_steps:
            # Update camera position based on sliders
            env.update_camera_position()

            # Get visual observation
            image = env.get_camera_image()
            # Convert image to PIL format for processor
            image = Image.fromarray(image).convert("RGB")

            # # Update the can's position
            # x = p.readUserDebugParameter(env.x_slider)
            # y = p.readUserDebugParameter(env.y_slider)
            # z = p.readUserDebugParameter(env.z_slider)
            # p.resetBasePositionAndOrientation(env.can_id, [x, y, z], [0, 0, 0, 1])

            # Get action from VLA model (simulate calling it once)
            # if step_count == 100 and not action_executed:  # Execute action after 100 steps
            #     target_action = get_action_from_openvla(image, prompt)

            #     # Execute the action using the robot
            #     env.robot.move_ee(target_action, control_method="end")
            #     action_executed = True
            #     print("Action executed!")

            # elif step_count == 300 and "grasp" in prompt or "pick up" in prompt:
            #     # Close gripper after moving to position
            #     env.robot.close_gripper()
            #     print("Gripper closed!")

            # elif step_count == 500 and ("grasp" in prompt or "pick up" in prompt):
            #     # Lift the can
            #     lift_action = [0.5, 0, 0.9, 0, math.pi / 2, 0]  # Lift up
            #     env.robot.move_ee(lift_action, control_method="end")
            #     print("Lifting can!")
            inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
            target_action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
            print(f"OpenVLA generated action is: {target_action}")
            new_pose, gripper_action = env.apply_incremental_action(target_action[:6])  # Use first 6 elements for pose
            print(f"New incremental action applied: {target_action[:6]}")
            print(f"New absolute pose: {new_pose}")

            # Execute the action
            env.robot.move_ee(new_pose, control_method="end")
            # env.robot.move_ee(gripper_action, control_method="end")
            # Step simulation
            env.step_simulation()
            step_count += 1

            # Optional: break early if user presses a key (you can implement this)
            # For now, just run for the specified steps

    except KeyboardInterrupt:
        print("Simulation interrupted by user")

    finally:
        # Clean up
        input("Press Enter to close the simulation...")
        env.close()


if __name__ == "__main__":
    can_grasping_demo()
