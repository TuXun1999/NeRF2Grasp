import sapien.core as sapien
from sapien.utils.viewer import Viewer
import numpy as np
import json
def demo(fix_root_link, balance_passive_force):
    engine = sapien.Engine()
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)

    scene_config = sapien.SceneConfig()
    scene = engine.create_scene(scene_config)
    scene.set_timestep(1 / 240.0)
    scene.add_ground(0)

    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

    viewer = Viewer(renderer)
    viewer.set_scene(scene)
    viewer.set_camera_xyz(x=-2, y=0, z=1)
    viewer.set_camera_rpy(r=0, p=-0.3, y=0)

    # Load URDF
    loader: sapien.URDFLoader = scene.create_urdf_loader()
    loader.fix_root_link = fix_root_link
    robot: sapien.Articulation = loader.load("./Sapien/test_data/SPOT/urdf/SPOT.urdf")
    robot.set_root_pose(sapien.Pose([0, 0, 0.48], [1, 0, 0, 0]))

    # Set initial joint positions
    x = robot.get_joints()
    robot_stand_data_dir = "./Sapien/robot_stand_data.json"
    with open(robot_stand_data_dir) as json_file:
        robot_stand_data = json.load(json_file)
    init_qpos = list(np.zeros(19))
    idx = 0
    for _, value in robot_stand_data.items():
        
        init_qpos[idx] = value
        idx = idx + 1
    robot.set_qpos(init_qpos)

    use_internal_PID = True
    if use_internal_PID == True:
        active_joints = robot.get_active_joints()
        target_qpos = init_qpos
        for joint_idx, joint in enumerate(active_joints):
            joint.set_drive_property(stiffness=20, damping=5)
            joint.set_drive_target(target_qpos[joint_idx])
    
    while not viewer.closed:
        for _ in range(4):  # render every 4 steps; TODO: correct
            if balance_passive_force:
                qf = robot.compute_passive_force(
                    gravity=True, 
                    coriolis_and_centrifugal=True 
                )
                robot.set_qf(qf)
            scene.step()
        scene.update_render()
        viewer.render()

    
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fix-root-link', action='store_true')
    parser.add_argument('--balance-passive-force', action='store_true')
    args = parser.parse_args()

    demo(fix_root_link=args.fix_root_link,
         balance_passive_force=args.balance_passive_force)


if __name__ == '__main__':
    main()
