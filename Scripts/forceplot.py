
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
np.set_printoptions(suppress=True)

def plot_error_bars(error_z_list, error_y_list, save_path=None):
    """
    Plot a 2x2 bar chart comparing Z and Y direction errors.

    Parameters:
    - error_z_list: list of 4 lists, each with 5 values (for Z direction)
    - error_y_list: list of 4 lists, each with 5 values (for Y direction)
    - save_path: str or None, if provided, saves the figure to this path
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    subplot_order = [(0, 1), (0, 0), (1, 0), (1, 1)]
    labels = ['normal', 'rf_loss', 'lf_loss', 'lh_loss', 'rh_loss']
    titles = ['RF Error', 'LF Error', 'LH Error', 'RH Error']

    x = np.arange(len(labels))  # [0, 1, 2, 3, 4]
    width = 0.35  # bar width

    for i in range(4):
        ax = axes[subplot_order[i]]
        z_vals = error_z_list[i]
        y_vals = error_y_list[i]

        ax.bar(x - width/2, z_vals, width, label='Z error')
        ax.bar(x + width/2, y_vals, width, label='Y error')

        ax.set_title(titles[i])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45)
        ax.set_ylabel('Mean Abs Error')
        ax.set_ylim(0, max(max(z_vals), max(y_vals)) * 1.2)
        ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
def plot_prediction_signals(esn_predict, mlp_predict_y, mlp_predict_z, target, num_samples=10):
    """
    Plot 10 random samples on a shared timeline:
    - Target: full cycle (140 steps), vivid green, thick and transparent
    - ESN: full cycle (140 steps), blue
    - MLP: latter half only (70–139), red dashed and thick

    Parameters:
    - esn_predict: (n_samples, 140, 2)
    - mlp_predict_y: (n_samples, 70)
    - mlp_predict_z: (n_samples, 70)
    - target: (n_samples, 140, 2)
    """
    n_samples = esn_predict.shape[0]
    sample_indices = np.random.choice(n_samples, size=num_samples, replace=False)

    fig, axs = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle("Target (full), ESN (full), MLP (latter half)", fontsize=16)

    # Fixed color styles
    color_target = 'green'
    color_esn = 'blue'
    color_mlp = 'red'

    for i, idx in enumerate(sample_indices):
        start = i * 140
        end = start + 140
        mid = start + 70
        time_full = np.arange(start, end)
        time_second_half = np.arange(mid, end)

        # Z signal
        axs[0].plot(time_full, target[idx, :, 0],
                    color=color_target, alpha=0.4, linewidth=2, label='Target Z' if i == 0 else "")
        axs[0].plot(time_full, esn_predict[idx, :, 0],
                    color=color_esn, alpha=0.8, label='ESN Predict Z' if i == 0 else "")
        axs[0].plot(time_second_half, mlp_predict_z[idx],
                    color=color_mlp, linestyle='--', alpha=0.9, linewidth=2.5,
                    label='MLP Predict Z (latter)' if i == 0 else "")

        # Y signal
        axs[1].plot(time_full, target[idx, :, 1],
                    color=color_target, alpha=0.4, linewidth=2, label='Target Y' if i == 0 else "")
        axs[1].plot(time_full, esn_predict[idx, :, 1],
                    color=color_esn, alpha=0.8, label='ESN Predict Y' if i == 0 else "")
        axs[1].plot(time_second_half, mlp_predict_y[idx],
                    color=color_mlp, linestyle='--', alpha=0.9, linewidth=2.5,
                    label='MLP Predict Y (latter)' if i == 0 else "")

    axs[0].set_title("Z Signal")
    axs[1].set_title("Y Signal")

    for ax in axs:
        ax.grid(True)
        ax.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

def plot_grf_prediction(test_set, grf_predicted, start=0, end=None):
    """
    Plot the true and predicted GRF values for Z and Y directions within a specified range.

    Parameters:
    - test_set: np.ndarray, shape (n_samples, 2), the ground truth values
    - grf_predicted: np.ndarray, shape (n_samples, 2), the predicted values
    - start: int, starting index for plotting
    - end: int or None, ending index for plotting (exclusive); if None, plots to the end
    """
    if end is None:
        end = test_set.shape[0]

    plt.close('all')
    
    plt.subplot(2, 1, 1)
    plt.plot(test_set[start:end, 0], label='true', alpha=0.5)
    plt.plot(grf_predicted[start:end, 0], label='predict', alpha=0.5)
    plt.legend(loc='upper right')
    plt.title('grf_Z_predicted')
    
    plt.subplot(2, 1, 2)
    plt.plot(test_set[start:end, 1], label='true', alpha=0.5)
    plt.plot(grf_predicted[start:end, 1], label='predict', alpha=0.5)
    plt.legend(loc='upper right')
    plt.title('grf_Y_predicted')
    
    plt.tight_layout()
    plt.show()


def plot_joints(joint_angles, title="Joint Angles", xlabel="Time", ylabel="Angle (degrees)"):
    """
    绘制四关节角度信号
    
    参数:
        joint_angles: 形状为(n,4)的numpy数组，包含四个关节的角度数据
        title: 图表标题 (默认: "Joint Angles")
        xlabel: x轴标签 (默认: "Time")
        ylabel: y轴标签 (默认: "Angle (degrees)")
    """
    # 检查输入数据
    if not isinstance(joint_angles, np.ndarray) or joint_angles.shape[1] != 4:
        raise ValueError("输入数据必须是形状为(n,4)的numpy数组")
    
    # 创建图形和轴
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 时间轴 (假设是等间隔采样)
    time = np.arange(joint_angles.shape[0])
    
    # 为每个关节绘制曲线
    joints = ['Joint 1', 'Joint 2', 'Joint 3', 'Joint 4']
    colors = ['b', 'g', 'r', 'c']  # 不同颜色区分关节
    
    for i in range(4):
        ax.plot(time, joint_angles[:, i], 
                color=colors[i], 
                label=joints[i],
                linewidth=1.5)
    
    # 添加图例和标签
    ax.legend(loc='upper right')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # # 添加网格
    # ax.grid(True, linestyle='--', alpha=0.6)
    
    # 自动调整布局
    plt.tight_layout()
    
    # 显示图形


def plot_legs(data, joint_label='关节', title='四足机器人关节数据', save_path=None):
    """
    绘制四足机器人关节数据的2x2子图
    
    参数:
        data: numpy数组，形状为[n,16]，包含四个腿的关节数据
        joint_label: 关节标签名称 (默认: '关节')
        title: 图表标题 (默认: '四足机器人关节数据')
        save_path: 图片保存路径 (可选)，如'output.png'
    """
    # 设置中文字体为宋体
    plt.rcParams['font.sans-serif'] = ['SimSun']  # 或者使用 'STSong'
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    # 检查数据形状
    if data.shape[1] != 16:
        raise ValueError("输入数据的列数应为16 (4腿×4关节)")
    
    # 创建2x2的子图布局
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(title, fontsize=16)

    # 定义每个腿的标签和颜色
    leg_names = ['腿1', '腿2', '腿3', '腿4']
    joint_names = [f'{joint_label}{i+1}' for i in range(4)]  # 使用输入的关节标签
    colors = ['r', 'g', 'b', 'm']  # 每种关节的颜色
    line_styles = ['-', '--', ':', '-.']  # 可选的线型

    # 绘制每个腿的数据
    for leg_idx in range(4):
        # 确定子图位置
        if leg_idx == 0:  # Leg1 -> (0,1)
            ax = axs[0, 1]
        elif leg_idx == 1:  # Leg2 -> (0,0)
            ax = axs[0, 0]
        elif leg_idx == 2:  # Leg3 -> (1,0)
            ax = axs[1, 0]
        else:               # Leg4 -> (1,1)
            ax = axs[1, 1]
        
        # 提取当前腿的数据 (4个关节)
        leg_data = data[:, leg_idx*4 : (leg_idx+1)*4]
        
        # 绘制每个关节的数据
        for joint_idx in range(4):
            ax.plot(leg_data[:, joint_idx], 
                    color=colors[joint_idx],
                    linestyle=line_styles[joint_idx],
                    linewidth=1.5,
                    label=joint_names[joint_idx])
        
        # 设置子图标题和标签
        ax.set_title(f'{leg_names[leg_idx]}{joint_label}数据')
        ax.set_xlabel('时间点')
        ax.set_ylabel(f'{joint_label}值(rad)')
        ax.legend(loc='upper right')
        # ax.grid(True, linestyle='--', alpha=0.6)

    # 调整子图间距
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    # 保存或显示图形
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    plt.show()

    
def plot_force_sensors(data, title='力传感器数据', save_path=None):
    """
    绘制四个力传感器的Y/Z方向力学信号的2x2子图
    
    参数:
        data: numpy数组，形状为[n,16]，
        title: 图表标题 (默认: '力传感器数据')
        save_path: 图片保存路径 (可选)
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimSun']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 检查数据形状
    if data.shape[1] != 8:
        raise ValueError("输入数据列数应为86 (4传感器×2方向)")
    
    # 创建2x2子图
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(title, fontsize=16)

    # 传感器配置 (名称, 子图位置, 数据列索引)
    sensor_config = [
        ('FS2', axs[0, 1], (0, 1)),   # 右上: FS2 (Y:0列, Z:1列)
        ('FS6', axs[0, 0], (2, 3)),   # 右上: FS6 (Y:2列, Z:3列)
        ('FS4', axs[1, 0], (4, 5)),   # 左下: FS4 (Y:4列, Z:5列)
        ('FS3', axs[1, 1], (6, 7))    # 右下: FS3 (Y:6列, Z:7列)
    ]

    # 绘制每个传感器的数据
    for name, ax, (y_col, z_col) in sensor_config:
        y_data = data[:, y_col]
        z_data = data[:, z_col]
        time = np.arange(len(y_data))
        
        ax.plot(time, y_data, 'b-', label='Z方向力')
        ax.plot(time, z_data, 'r-', label='Y方向力')
        ax.set_title(name)
        ax.set_xlabel('时间点')
        ax.set_ylabel('力(N)')
        ax.set_ylim(-4, 4)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    # 保存或显示
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    
    plt.show()

