from autolife_planning.dataclass.robot_description import CameraConfig, RobotConfig

autolife_robot_config = RobotConfig(
    urdf_path="/media/run/Extend/Autolife-Planning/assets/robot/autolife/autolife.urdf",
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
