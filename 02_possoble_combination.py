from  kinematics_utils import kinematics
import numpy as np


kin_model=kinematics_model(ur_model='ur5', gripper_offset=0.15)

init_joints=[-0.0962801,  -2.3614519,   3.0718334,   0.8604148,   1.5707963,  -1.4745163 ]
pose=[0.1, 0.1, 0]
ee_orientation=np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
joints_ik=kin_model.get_joint_combination(pose, ee_orientation, init_joints)

