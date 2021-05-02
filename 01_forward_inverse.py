from  kinematics_utils import kinematics
import numpy as np


kin_model=kinematics.kinematics_model(ur_model='ur5', gripper_offset=0.15)


joints=[np.pi, 0, 0, 0, 0, 0]
pose, ee_orientation=kin_model.forward_kin(joints)
print(pose)
#print(ee_orientation)
joints_ik=kin_model.inverse_kin(pose, ee_orientation)

#tests all possibilities
for i in range(joints_ik.shape[0]):
    new_pose, new_ee_orientation=kin_model.forward_kin(joints_ik[i, :])
    print(new_pose)
    #print(new_ee_orientation)

    #print((np.abs(new_ee_orientation-ee_orientation) > 0.0001).any())
    if (np.abs(new_ee_orientation-ee_orientation) > 0.001).any():
        print(new_ee_orientation-ee_orientation)