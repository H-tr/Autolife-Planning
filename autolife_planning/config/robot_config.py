import os

from autolife_planning.types.robot import CameraConfig, ChainConfig, RobotConfig

_RESOURCES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "resources",
    "robot",
    "autolife",
)

CHAIN_CONFIGS: dict[str, ChainConfig] = {
    "left_arm": ChainConfig(
        base_link="Link_Waist_Yaw_to_Shoulder_Inner",
        ee_link="Link_Left_Wrist_Lower_to_Gripper",
        num_joints=7,
        urdf_path=os.path.join(_RESOURCES_DIR, "autolife.urdf"),
    ),
    "right_arm": ChainConfig(
        base_link="Link_Waist_Yaw_to_Shoulder_Inner",
        ee_link="Link_Right_Wrist_Lower_to_Gripper",
        num_joints=7,
        urdf_path=os.path.join(_RESOURCES_DIR, "autolife.urdf"),
    ),
    "whole_body_left": ChainConfig(
        base_link="Link_Ground_Vehicle_Z",
        ee_link="Link_Left_Wrist_Lower_to_Gripper",
        num_joints=11,
        urdf_path=os.path.join(_RESOURCES_DIR, "autolife.urdf"),
    ),
    "whole_body_right": ChainConfig(
        base_link="Link_Ground_Vehicle_Z",
        ee_link="Link_Right_Wrist_Lower_to_Gripper",
        num_joints=11,
        urdf_path=os.path.join(_RESOURCES_DIR, "autolife.urdf"),
    ),
    "whole_body_base_left": ChainConfig(
        base_link="Link_Zero_Point",
        ee_link="Link_Left_Wrist_Lower_to_Gripper",
        num_joints=14,
        urdf_path=os.path.join(_RESOURCES_DIR, "autolife_base.urdf"),
    ),
    "whole_body_base_right": ChainConfig(
        base_link="Link_Zero_Point",
        ee_link="Link_Right_Wrist_Lower_to_Gripper",
        num_joints=14,
        urdf_path=os.path.join(_RESOURCES_DIR, "autolife_base.urdf"),
    ),
}

autolife_robot_config = RobotConfig(
    urdf_path=os.path.join(_RESOURCES_DIR, "autolife.urdf"),
    joint_names=[
        "Joint_Ankle",
        "Joint_Knee",
        "Joint_Waist_Pitch",
        "Joint_Waist_Yaw",
        "Joint_Left_Shoulder_Inner",
        "Joint_Left_Shoulder_Outer",
        "Joint_Left_UpperArm",
        "Joint_Left_Elbow",
        "Joint_Left_Forearm",
        "Joint_Left_Wrist_Upper",
        "Joint_Left_Wrist_Lower",
        "Joint_Right_Shoulder_Inner",
        "Joint_Right_Shoulder_Outer",
        "Joint_Right_UpperArm",
        "Joint_Right_Elbow",
        "Joint_Right_Forearm",
        "Joint_Right_Wrist_Upper",
        "Joint_Right_Wrist_Lower",
    ],
    camera=CameraConfig(
        link_name="Link_Camera_Chest",
        width=640,
        height=480,
        fov=60.0,
        near=0.1,
        far=10.0,
    ),
)

HOME_JOINTS = [
    0.0887,
    -0.0293,
    -0.0128,
    -0.0424,
    -0.0700,
    0.1110,
    0.1234,
    -0.0488,
    -0.1148,
    -0.0954,
    0.1303,
    -0.1605,
    -0.2033,
    0.0565,
    -0.0134,
    -0.0278,
    0.1667,
    -0.0103,
]
