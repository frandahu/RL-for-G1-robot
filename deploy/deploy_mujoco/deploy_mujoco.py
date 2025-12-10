# import time

# import mujoco.viewer
# import mujoco
# import numpy as np
# from legged_gym import LEGGED_GYM_ROOT_DIR
# import torch
# import yaml


# # 引入键盘监听库
# try:
#     from pynput import keyboard
# except ImportError:
#     print("正在尝试安装 pynput...")
#     import subprocess
#     subprocess.check_call(["pip", "install", "pynput"])
#     from pynput import keyboard


# # ================= 控制参数配置 =================
# # 你可以在这里调整手感
# SPEED_X = 0.6      # 前进速度 (m/s)
# SPEED_BACK = -0.4  # 后退速度 (m/s)
# SPEED_Y = 0.3      # 横移速度 (m/s)
# SPEED_TURN = 0.8   # 转向速度 (rad/s)
# # ==============================================

# # 定义全局变量用于存储当前指令
# # [x_vel, y_vel, yaw_vel]
# current_cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)

# # 键盘状态记录
# class KeyState:
#     def __init__(self):
#         self.up = False
#         self.down = False
#         self.left = False
#         self.right = False
#         self.shift = False # 用于配合左右键实现横移

# key_state = KeyState()

# def on_press(key):
#     global current_cmd
#     step_lin = 0.1  # 线速度步长
#     step_ang = 0.1  # 角速度步长
    
#     try:
#         if key == keyboard.Key.up:       # ↑ 前进
#             current_cmd[0] += step_lin
#         elif key == keyboard.Key.down:   # ↓ 后退
#             current_cmd[0] -= step_lin
#         elif key == keyboard.Key.left:   # ← 左横移
#             current_cmd[1] += step_lin
#         elif key == keyboard.Key.right:  # → 右横移
#             current_cmd[1] -= step_lin
#         elif key.char == 'q':            # Q 左转
#             current_cmd[2] += step_ang
#         elif key.char == 'e':            # E 右转
#             current_cmd[2] -= step_ang
#         elif key.char == 'k':            # K 急停
#             current_cmd[:] = 0.0
            
#     except AttributeError:
#         pass
    
#     # 打印当前指令
#     print(f"\r当前指令 -> X: {current_cmd[0]:.2f}, Y: {current_cmd[1]:.2f}, Yaw: {current_cmd[2]:.2f}", end="")

# def start_keyboard_listener():
#     listener = keyboard.Listener(on_press=on_press)
#     listener.start()


# def get_gravity_orientation(quaternion):
#     qw = quaternion[0]
#     qx = quaternion[1]
#     qy = quaternion[2]
#     qz = quaternion[3]

#     gravity_orientation = np.zeros(3)

#     gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
#     gravity_orientation[1] = -2 * (qz * qy + qw * qx)
#     gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

#     return gravity_orientation


# def pd_control(target_q, q, kp, target_dq, dq, kd):
#     """Calculates torques from position commands"""
#     return (target_q - q) * kp + (target_dq - dq) * kd


# if __name__ == "__main__":

#     #### 启动键盘监听线程
#     print("\n" + "="*50)
#     print("【控制说明】")
#     print("  ↑ (方向键上) : 前进")
#     print("  ↓ (方向键下) : 后退")
#     print("  ← (方向键左) : 左横移")
#     print("  → (方向键右) : 右横移")
#     print("  Q           : 左原地转向")
#     print("  E           : 右原地转向")
#     print("  K           : 急停 (速度归零)")
#     print("="*50 + "\n")
#     start_keyboard_listener()
#     ####

#     # get config file name from command line
#     import argparse

#     parser = argparse.ArgumentParser()
#     parser.add_argument("config_file", type=str, help="config file name in the config folder")
#     args = parser.parse_args()
#     config_file = args.config_file
#     with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}", "r") as f:
#         config = yaml.load(f, Loader=yaml.FullLoader)
#         policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
#         xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

#         simulation_duration = config["simulation_duration"]
#         simulation_dt = config["simulation_dt"]
#         control_decimation = config["control_decimation"]

#         kps = np.array(config["kps"], dtype=np.float32)
#         kds = np.array(config["kds"], dtype=np.float32)

#         default_angles = np.array(config["default_angles"], dtype=np.float32)

#         ang_vel_scale = config["ang_vel_scale"]
#         dof_pos_scale = config["dof_pos_scale"]
#         dof_vel_scale = config["dof_vel_scale"]
#         action_scale = config["action_scale"]
#         cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

#         num_actions = config["num_actions"]
#         num_obs = config["num_obs"]
        
#         ###### 初始化 current_cmd，优先使用 yaml 中的配置，但建议在 yaml 中设为 0
#         init_cmd_from_yaml = np.array(config["cmd_init"], dtype=np.float32)
#         current_cmd[:] = init_cmd_from_yaml[:]
#         ######

#     # define context variables
#     action = np.zeros(num_actions, dtype=np.float32)
#     target_dof_pos = default_angles.copy()
#     obs = np.zeros(num_obs, dtype=np.float32)

#     counter = 0

#     # Load robot model
#     m = mujoco.MjModel.from_xml_path(xml_path)
#     d = mujoco.MjData(m)
#     m.opt.timestep = simulation_dt

#     # load policy
#     policy = torch.jit.load(policy_path)

#     # push_interval = 2.0  # 每隔 3 秒推一次
#     # max_push_vel = 1.0   # 最大推动速度 (m/s)
#     # last_push_time = 0.5 # 上次推动的时间

#     with mujoco.viewer.launch_passive(m, d) as viewer:
#         # Close the viewer automatically after simulation_duration wall-seconds.
#         start = time.time()
#         while viewer.is_running() and time.time() - start < simulation_duration:
#             step_start = time.time()
#             tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
#             d.ctrl[:] = tau
#             # mj_step can be replaced with code that also evaluates
#             # a policy and applies a control signal before stepping the physics.
#             mujoco.mj_step(m, d)


        

#             # ================= [新增] 随机水平扰动逻辑 =================
#             # 检查当前仿真时间 d.time 是否超过了下一次推动时间
#             # if d.time - last_push_time >= push_interval:
#             #     last_push_time = d.time
                
#             #     # 生成 XY 平面的随机速度 (模拟来自不同方向的推力)
#             #     # d.qvel[0] 是 x 轴线速度, d.qvel[1] 是 y 轴线速度
#             #     push_vel_x = np.random.uniform(-max_push_vel, max_push_vel)
#             #     push_vel_y = np.random.uniform(-max_push_vel, max_push_vel)
                
#             #     # 直接叠加速度 (模拟瞬间冲击/Impulse)
#             #     d.qvel[0] += push_vel_x
#             #     d.qvel[1] += push_vel_y
                
#             #     print(f"[Sim2Sim] Time: {d.time:.2f}s | Pushed: vx={push_vel_x:.2f}, vy={push_vel_y:.2f}")
#             # # ==========================================================

#             counter += 1
#             if counter % control_decimation == 0:
#                 # Apply control signal here.

#                 # create observation
#                 qj = d.qpos[7:]
#                 dqj = d.qvel[6:]
#                 quat = d.qpos[3:7]
#                 omega = d.qvel[3:6]

#                 qj = (qj - default_angles) * dof_pos_scale
#                 dqj = dqj * dof_vel_scale
#                 gravity_orientation = get_gravity_orientation(quat)
#                 omega = omega * ang_vel_scale

#                 period = 0.8
#                 count = counter * simulation_dt
#                 phase = count % period / period
#                 sin_phase = np.sin(2 * np.pi * phase)
#                 cos_phase = np.cos(2 * np.pi * phase)

#                 obs[:3] = omega
#                 obs[3:6] = gravity_orientation
#                 # obs[6:9] = cmd * cmd_scale
#                 ###### 【修改点】这里使用全局变量 current_cmd 而不是静态的 config cmd
#                 obs[6:9] = current_cmd * cmd_scale
#                 ######
#                 obs[9 : 9 + num_actions] = qj
#                 obs[9 + num_actions : 9 + 2 * num_actions] = dqj
#                 obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action
#                 obs[9 + 3 * num_actions : 9 + 3 * num_actions + 2] = np.array([sin_phase, cos_phase])
#                 obs_tensor = torch.from_numpy(obs).unsqueeze(0)

#                 # 后面的代码不变，确保 cmd 被写入 obs
#                 obs[6:9] = current_cmd * cmd_scale 
#                 # policy inference
#                 action = policy(obs_tensor).detach().numpy().squeeze()
#                 # transform action to target_dof_pos
#                 target_dof_pos = action * action_scale + default_angles

#             # Pick up changes to the physics state, apply perturbations, update options from GUI.
#             viewer.sync()

#             # Rudimentary time keeping, will drift relative to wall clock.
#             time_until_next_step = m.opt.timestep - (time.time() - step_start)
#             if time_until_next_step > 0:
#                 time.sleep(time_until_next_step)




import time
import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml
import sys
import pygame


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]
    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation

def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd

if __name__ == "__main__":
    # 1. 初始化 Pygame (用于接收键盘输入)
    pygame.init()
    # 创建一个小窗口用于捕获焦点 (必须有窗口才能接收按键)
    screen = pygame.display.set_mode((400, 300))
    pygame.display.set_caption("G1 遥控器 (请点我聚焦)")
    
    # 在 Pygame 窗口上显示操作提示
    font = pygame.font.SysFont("Arial", 18)
    def draw_instructions(cmd_vel):
        screen.fill((30, 30, 30)) # 深灰背景
        texts = [
            "=== G1 Controller ===",
            "Use Arrow Keys to Control",
            f"Forward/Back: {cmd_vel[0]:.2f}",
            f"Strafe Left/Right: {cmd_vel[1]:.2f}", # 这里显示平移速度
            f"Turn Yaw: {cmd_vel[2]:.2f}",          # 这里显示转向速度
            "",
            "[Up/Down]: Forward / Backward",
            "[Left/Right]: Strafe Left / Right",    # 更新说明：直接左右键是平移
            "[Shift + Left/Right]: Turn Yaw",       # 更新说明：Shift+左右是转向
            "",
            "KEEP THIS WINDOW FOCUSED!"
        ]
        for i, line in enumerate(texts):
            color = (0, 255, 0) if "Cmd" not in line else (255, 255, 255)
            text_surf = font.render(line, True, (200, 200, 200))
            screen.blit(text_surf, (20, 20 + i * 25))
        pygame.display.flip()

    # 控制参数配置
    VEL_X_MAX = 0.6    # 前进速度
    VEL_BACK_MAX = 0.4 # 后退速度
    VEL_Y_MAX = 0.5    # 横移速度
    VEL_YAW_MAX = 1  # 转向速度

    # 读取配置文件
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    
    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{args.config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]
        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)
        default_angles = np.array(config["default_angles"], dtype=np.float32)
        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        
        # 初始化指令
        cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)
    counter = 0

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    policy = torch.jit.load(policy_path)

    print("\n==============================================")
    print("   G1 机器人 MuJoCo 控制器 (Pygame版)")
    print("   请确保点击弹出的【G1 遥控器】黑色窗口！")
    print("   ↑/↓ : 前进/后退")
    print("   ←/→ : 左右平移 (Strafe)")
    print("   Shift + ←/→ : 转向 (Turn)")
    print("==============================================\n")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()
        last_render = 0
        
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            
            # --- 物理步进 ---
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)
            counter += 1

            # --- 控制与推理 (50Hz) ---
            if counter % control_decimation == 0:
                # 1. 处理 Pygame 事件 (防止窗口卡死)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        viewer.close()
                        break
                
                # 2. 获取按键状态 (实时轮询，极度灵敏)
                keys = pygame.key.get_pressed()
                
                # 计算 X 轴速度
                target_vx = 0.0
                if keys[pygame.K_UP]: target_vx += VEL_X_MAX
                if keys[pygame.K_DOWN]: target_vx -= VEL_BACK_MAX
                
                # 计算 Y 轴和 Yaw 轴 (按住Shift切换)
                target_vy = 0.0
                target_dyaw = 0.0
                is_shift = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
                
                if keys[pygame.K_LEFT]:
                    if is_shift: 
                        target_dyaw = VEL_YAW_MAX       # 【修改】Shift+左 = 左转
                    else: 
                        target_vy = VEL_Y_MAX           # 【修改】纯左键 = 左平移
                
                if keys[pygame.K_RIGHT]:
                    if is_shift: 
                        target_dyaw = -VEL_YAW_MAX      # 【修改】Shift+右 = 右转
                    else: 
                        target_vy = -VEL_Y_MAX          # 【修改】纯右键 = 右平移
                
                # 平滑更新指令 (简单的线性插值，让启动不那么突兀)
                # alpha = 0.2 # 0~1, 越大反应越快
                # cmd = cmd * (1-alpha) + np.array([target_vx, target_vy, target_dyaw]) * alpha
                # 或者直接赋值实现最快响应：
                cmd = np.array([target_vx, target_vy, target_dyaw], dtype=np.float32)

                # 3. 刷新 Pygame 界面显示 (降低刷新率以节省性能)
                if time.time() - last_render > 0.1:
                    draw_instructions(cmd)
                    last_render = time.time()

                # 4. 机器人状态观测与推理
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]
                omega = d.qvel[3:6]
                qj = (qj - default_angles) * dof_pos_scale
                dqj = dqj * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)
                omega = omega * ang_vel_scale
                period = 0.8
                count = counter * simulation_dt
                phase = count % period / period
                sin_phase = np.sin(2 * np.pi * phase)
                cos_phase = np.cos(2 * np.pi * phase)

                obs[:3] = omega
                obs[3:6] = gravity_orientation
                obs[6:9] = cmd * cmd_scale  # 注入控制指令
                obs[9 : 9 + num_actions] = qj
                obs[9 + num_actions : 9 + 2 * num_actions] = dqj
                obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action
                obs[9 + 3 * num_actions : 9 + 3 * num_actions + 2] = np.array([sin_phase, cos_phase])
                
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                action = policy(obs_tensor).detach().numpy().squeeze()
                target_dof_pos = action * action_scale + default_angles

            viewer.sync()
            
            # 简单的时间同步
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    
    pygame.quit()
