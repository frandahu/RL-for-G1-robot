from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class G1RoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.8] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
           'left_hip_yaw_joint' : 0. ,   
           'left_hip_roll_joint' : 0,               
           'left_hip_pitch_joint' : -0.1,         
           'left_knee_joint' : 0.3,       
           'left_ankle_pitch_joint' : -0.2,     
           'left_ankle_roll_joint' : 0,     
           'right_hip_yaw_joint' : 0., 
           'right_hip_roll_joint' : 0, 
           'right_hip_pitch_joint' : -0.1,                                       
           'right_knee_joint' : 0.3,                                             
           'right_ankle_pitch_joint': -0.2,                              
           'right_ankle_roll_joint' : 0,       
           'torso_joint' : 0.
        }
    
    class env(LeggedRobotCfg.env):
        num_observations = 47
        num_privileged_obs = 50
        num_actions = 12


    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        push_robots = True
        push_interval_s = 3
        max_push_vel_xy = 1.5
        # 模拟 0-6 个 step 的延迟
        randomize_lag_timesteps = True
        lag_timesteps = 6 
      

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
          # PD Drive parameters:
        stiffness = {'hip_yaw': 100,
                     'hip_roll': 100,
                     'hip_pitch': 100,
                     'knee': 150,
                     'ankle': 40,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 2,
                     'hip_roll': 2,
                     'hip_pitch': 2,
                     'knee': 4,
                     'ankle': 2,
                     }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_12dof.urdf'
        name = "g1"
        foot_name = "ankle_roll"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
    
    class commands( LeggedRobotCfg.commands ):
        class ranges:
            lin_vel_x = [-0.5, 0.5]   # 恢复移动能力，但限制最大速度以保安全
            lin_vel_y = [-0.3, 0.3]
            ang_vel_yaw = [-0.5, 0.5]
            heading = [-0.0, 0.0]
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.78
        only_positive_rewards = True
        
        class scales( LeggedRobotCfg.rewards.scales ):
            tracking_lin_vel = 1.5  #1.0
            tracking_ang_vel = 1 #0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            base_height = -10.0
            dof_acc = -2.5e-7 * 10
            dof_vel = -1e-3
            feet_air_time = 0.0
            collision = 0.0
            action_rate = -0.01 * 5
            dof_pos_limits = -5.0
            #alive = 0.15
            hip_pos = -1.0
            contact_no_vel = -0.2
            #feet_swing_height = -20.0
            #contact = 0.18

            # === 1. 禁用行走的奖励 ===
            contact = 0.0             # 【修改】原为 0.18，设为0禁用步态相位奖励
            feet_swing_height = 0.0   # 【修改】原为 -20.0，设为0允许脚贴地
            alive = 2
            # === 2. 启用静止的奖励 ===
            # 注意：stand_still 在父类中存在但权重为0，这里我们给它赋值
            stand_still = -1       # 【新增/覆写】强力惩罚非指令下的动作
            
            # === 3. 姿态约束 ===
            # 如果你想约束所有关节，stand_still 已经够了。
            # 如果你想特别约束髋关节，可以加强 hip_pos
            hip_pos = -1.0            # 【修改】或者设为更低，如 -2.0

class G1RoughCfgPPO( LeggedRobotCfgPPO ):
    class policy:
        init_noise_std = 0.8
        actor_hidden_dims = [32]
        critic_hidden_dims = [32]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'
        rnn_hidden_size = 64
        rnn_num_layers = 1
        
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = "ActorCriticRecurrent"
        max_iterations = 10000
        run_name = ''
        experiment_name = 'g1'

#python deploy/deploy_mujoco/deploy_mujoco.py g1.yaml

#python deploy/deploy_real/deploy_real.py enp7s0 g1.yaml


