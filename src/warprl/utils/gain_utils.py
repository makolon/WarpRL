def print_gain(model):
    print("joint_limit_ke = ", model.joint_limit_ke)
    print("joint_limit_kd = ", model.joint_limit_kd)
    print("joint_ke = ", model.joint_target_ke)
    print("joint_kd = ", model.joint_target_kd)
    print("joint_ke shape = ", model.joint_target_ke.shape)
    print("joint_kd shape = ", model.joint_target_kd.shape)
    print("joint_limit_lower = ", model.joint_limit_lower)
    print("joint_limit_upper = ", model.joint_limit_upper)
    input("Press Enter to continue...")
