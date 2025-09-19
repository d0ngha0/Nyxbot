
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.ndimage import gaussian_filter1d
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MaxNLocator
np.set_printoptions(suppress=True)



colors = ["#A66A81", "#F4663A", "#8A7CCB", "#A9CF8E", "#2FAEC7", "#E4E9ED"]


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


# def plot_prediction_signals(esn_predict, mlp_predict_y, mlp_predict_z, target, save_path=None):
#             """
#             Plot up to num_samples cycles in order (from 0 to num_samples-1):
#             - Target: full cycle (140 steps), vivid green, thick and transparent
#             - ESN: full cycle (140 steps), blue
#             - MLP: latter half only (70–139), red dashed and thick, for each cycle

#             Parameters:
#             - esn_predict: (n*140, 2)
#             - mlp_predict_y: (n, 70)
#             - mlp_predict_z: (n, 70)
#             - target: (n*140, 2)
#             - num_samples: number of cycles to plot (in order from 0)
#             """
#             n_cycles = mlp_predict_y.shape[0]
#             assert esn_predict.shape[0] == target.shape[0] == n_cycles * 140, "Mismatch in data shapes"

#             indices = range(n_cycles)

#             fig, axs = plt.subplots(2, 1, figsize=(10, 6))

#             color_target = 'green'
#             color_esn = 'blue'
#             color_mlp = 'red'

#             max_data_len = n_cycles * 140

#             for i, idx in enumerate(indices):
#                 start = idx * 140
#                 end = start + 140
#                 mid = start + 70
#                 time_full = np.arange(start, end)
#                 time_second_half = np.arange(mid, end)

#                 # Z signal
#                 axs[0].plot(time_full, target[start:end, 0],
#                             color=color_target, alpha=0.4, linewidth=6, label='Real Z' if i == 0 else "")
#                 axs[0].plot(time_full, esn_predict[start:end, 0],
#                             color=color_esn, alpha=0.8, linewidth=4,label='ESN prediction' if i == 0 else "")
#                 axs[0].plot(time_second_half, mlp_predict_z[idx],
#                             color=color_mlp,  alpha=0.9, linewidth=5,
#                             label='MLP prediction' if i == 0 else "")

#                 # Y signal
#                 axs[1].plot(time_full, target[start:end, 1],
#                             color=color_target, alpha=0.4, linewidth=6, label='Real Y' if i == 0 else "")
#                 axs[1].plot(time_full, esn_predict[start:end, 1],
#                             color=color_esn, alpha=0.8, linewidth=4, label='ESN prediction' if i == 0 else "")
#                 axs[1].plot(time_second_half, mlp_predict_y[idx],
#                             color=color_mlp,  alpha=0.9, linewidth=5,
#                             label='MLP prediction' if i == 0 else "")

#             axs[0].set_ylabel("GRF_Z", fontsize=14)
#             axs[1].set_ylabel("GRF_Y", fontsize=14)
#             axs[1].set_xlabel("Time (step)", fontsize=14)

#             axs[0].set_xticks([])  # Remove x ticks on upper subplot

#             axs[0].set_xlim(0, max_data_len)
#             axs[1].set_xlim(0, max_data_len)

#             for ax in axs:
#                 # ax.grid(True)
#                 ax.legend(loc='upper right',fontsize=12)

#             plt.tight_layout()
#             if save_path:
#                 plt.savefig(save_path, dpi=300, bbox_inches='tight')
#                 print(f"Figure saved to {save_path}")
#             else:
#                 plt.show()
def plot_prediction_signals(esn_predict, mlp_predict_y, mlp_predict_z, target, save_path=None,
                          label_size=25, legend_size=16, tick_size=14, 
                          line_width_target=6, line_width_esn=4, line_width_mlp=5):
    """
    Plot up to num_samples cycles in order (from 0 to num_samples-1):
    - Target: full cycle (140 steps), vivid green, thick and transparent
    - ESN: full cycle (140 steps), blue
    - MLP: latter half only (70-139), red dashed and thick, for each cycle

    Parameters:
    - esn_predict: (n*140, 2)
    - mlp_predict_y: (n, 70)
    - mlp_predict_z: (n, 70)
    - target: (n*140, 2)
    - save_path: str or None, path to save figure
    - label_size: int, font size for axis labels
    - legend_size: int, font size for legend text
    - tick_size: int, font size for axis ticks
    - line_width_target: int, line width for target signal
    - line_width_esn: int, line width for ESN prediction
    - line_width_mlp: int, line width for MLP prediction
    """
    n_cycles = mlp_predict_y.shape[0]
    assert esn_predict.shape[0] == target.shape[0] == n_cycles * 140, "Mismatch in data shapes"

    indices = range(n_cycles)

    fig, axs = plt.subplots(2, 1, figsize=(10, 6))

    color_target = 'green'
    color_esn = 'blue'
    color_mlp = 'red'

    max_data_len = n_cycles * 140

    for i, idx in enumerate(indices):
        start = idx * 140
        end = start + 140
        mid = start + 70
        time_full = np.arange(start, end)
        time_second_half = np.arange(mid, end)

        # Z signal
        axs[0].plot(time_full, target[start:end, 0],
                    color=color_target, alpha=0.4, linewidth=line_width_target, 
                    label='Real Z' if i == 0 else "")
        axs[0].plot(time_full, esn_predict[start:end, 0],
                    color=color_esn, alpha=0.8, linewidth=line_width_esn,
                    label='ESN prediction' if i == 0 else "")
        axs[0].plot(time_second_half, mlp_predict_z[idx],
                    color=color_mlp, alpha=0.9, linewidth=line_width_mlp,
                    label='MLP prediction' if i == 0 else "")

        # Y signal
        axs[1].plot(time_full, target[start:end, 1],
                    color=color_target, alpha=0.4, linewidth=line_width_target, 
                    label='Real Y' if i == 0 else "")
        axs[1].plot(time_full, esn_predict[start:end, 1],
                    color=color_esn, alpha=0.8, linewidth=line_width_esn,
                    label='ESN prediction' if i == 0 else "")
        axs[1].plot(time_second_half, mlp_predict_y[idx],
                    color=color_mlp, alpha=0.9, linewidth=line_width_mlp,
                    label='MLP prediction' if i == 0 else "")

    # Apply consistent styling
    for ax in axs:
        ax.tick_params(axis='both', which='major', labelsize=tick_size)
        ax.legend(loc='upper right', fontsize=legend_size)
        ax.set_xlim(0, max_data_len)

    # Set y-ticks and left-aligned y-labels
    axs[0].set_yticks([-10, -5, 0, 5, 10])
    axs[1].set_yticks([-2, 2, 6, 10])
    
    # Left-align y-axis labels
    axs[0].set_ylabel("GRA_Z (N)", fontsize=label_size)
    axs[1].set_ylabel("GRA_Y (N)", fontsize=label_size)
    
    # Adjust y-label positions
    axs[0].yaxis.set_label_coords(-0.05, 0.5)  # (x, y) coordinates (-0.05, 0.5)
    axs[1].yaxis.set_label_coords(-0.05, 0.5)
    
    axs[1].set_xlabel("Time (step)", fontsize=label_size)
    axs[0].set_xticks([])  # Remove x ticks on upper subplot

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()



def plot_grf_prediction(test_set, grf_predicted, start=0, end=None, save_path=None,
                       label_size=25, legend_size=16, tick_size=14, line_width_pred=4, line_width_real=5):
    """
    Plot the true and predicted GRF values for Z and Y directions within a specified range.

    Parameters:
    - test_set: np.ndarray, shape (n_samples, 2), the ground truth values
    - grf_predicted: np.ndarray, shape (n_samples, 2), the predicted values
    - start: int, starting index for plotting
    - end: int or None, ending index for plotting (exclusive); if None, plots to the end
    - save_path: str or None, if provided, saves the figure to this path as a high-resolution PNG
    - label_size: int, font size for axis labels
    - legend_size: int, font size for legend
    - tick_size: int, font size for y-axis ticks
    - line_width_pred: int, line width for predicted values
    - line_width_real: int, line width for real values
    """
    if end is None:
        end = test_set.shape[0]

    data_len = end - start

    plt.close('all')
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    
    # Common plot settings
    plot_settings = {
        'label_size': label_size,
        'legend_size': legend_size,
        'tick_size': tick_size,
        'line_width_pred': line_width_pred,
        'line_width_real': line_width_real,
        'labelpad': 20,
        'y_label_coords': (-0.05, 0.5)
    }

    # Plot for GRF_Z
    axs[0].plot(test_set[start:end, 0], label='Real', color='blue', alpha=0.8, linewidth=plot_settings['line_width_real'])
    axs[0].plot(grf_predicted[start:end, 0], label='ESN prediction', color='red', alpha=1, linewidth=plot_settings['line_width_pred'])
    axs[0].legend(loc='upper right', fontsize=plot_settings['legend_size'])
    axs[0].set_ylabel('GRA_Z', fontsize=plot_settings['label_size'], labelpad=plot_settings['labelpad'])
    axs[0].set_xticks([])
    axs[0].set_yticks([-10,-5,0,5,10])
    axs[0].set_xlim(0, data_len)
    axs[0].tick_params(axis='y', which='major', labelsize=plot_settings['tick_size'])
    axs[0].yaxis.set_label_coords(*plot_settings['y_label_coords'])

    # Plot for GRF_Y
    axs[1].plot(test_set[start:end, 1], label='Real', color='blue', alpha=0.8, linewidth=plot_settings['line_width_real'])
    axs[1].plot(grf_predicted[start:end, 1], label='ESN prediction', color='red', alpha=1, linewidth=plot_settings['line_width_pred'])
    axs[1].legend(loc='upper right', fontsize=plot_settings['legend_size'])
    axs[1].set_ylabel('GRA_Y', fontsize=plot_settings['label_size'], labelpad=plot_settings['labelpad'])
    axs[1].set_xticks([])
    axs[1].set_yticks([-2,2,6,10])
    axs[1].tick_params(axis='y', which='major', labelsize=plot_settings['tick_size'])
    axs[1].set_xlim(0, data_len)
    axs[1].yaxis.set_label_coords(*plot_settings['y_label_coords'])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


# def plot_grf_prediction(test_set, grf_predicted, start=0, end=None, save_path=None):
#     """
#     Plot the true and predicted GRF values for Z and Y directions within a specified range.

#     Parameters:
#     - test_set: np.ndarray, shape (n_samples, 2), the ground truth values
#     - grf_predicted: np.ndarray, shape (n_samples, 2), the predicted values
#     - start: int, starting index for plotting
#     - end: int or None, ending index for plotting (exclusive); if None, plots to the end
#     - save_path: str or None, if provided, saves the figure to this path as a high-resolution PNG
#     """
#     if end is None:
#         end = test_set.shape[0]

#     data_len = end - start

#     plt.close('all')
#     fig, axs = plt.subplots(2, 1, figsize=(10, 6))
#     # Align y-labels for both subplots
#     axs[0].plot(test_set[start:end, 0], label='Real', color='blue', alpha=0.8, linewidth=5)
#     axs[0].plot(grf_predicted[start:end, 0], label='ESN prediction', color='red', alpha=1, linewidth=4)
#     axs[0].legend(loc='upper right', fontsize=12)
#     axs[0].set_ylabel('GRF_Z', fontsize=14, labelpad=20)
#     axs[0].set_xticks([])
#     axs[0].set_xlim(0, data_len)
#     axs[0].tick_params(axis='y', which='major', labelsize=14)
#     axs[0].yaxis.set_label_coords(-0.05, 0.5)  # Move y-label left for alignment

#     axs[1].plot(test_set[start:end, 1], label='Real', color='blue', alpha=0.8, linewidth=5)
#     axs[1].plot(grf_predicted[start:end, 1], label='ESN prediction', color='red', alpha=1, linewidth=4)
#     axs[1].legend(loc='upper right', fontsize=12)
#     axs[1].set_ylabel('GRF_Y', fontsize=14, labelpad=20)
#     # axs[1].set_xlabel('Time (step)', fontsize=14)
#     axs[1].set_xticks([])
#     axs[1].tick_params(axis='y', which='major', labelsize=14)
#     axs[1].set_xlim(0, data_len)
#     axs[1].yaxis.set_label_coords(-0.05, 0.5)  # Move y-label left for alignment

#     plt.tight_layout()
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         print(f"Figure saved to {save_path}")
#     else:
#         plt.show()


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


def plot_conf_matrix(data1, data2, data3, data4,
                    fmt=".2f", cmap="Blues", save_path=None):
    """
    Plots a 4×2 matrix for each limb (rows: RF, LF, LH, RH) and axis
    (columns: GRF_Y, GRF_Z). Each cell shows the value from the provided
    data arrays, and you can save the figure by passing save_path.

    Parameters
    ----------
    data1–4 : np.array of shape (2,) or (2,1)
        [GRF_Y, GRF_Z] values for RF, LF, LH, RH respectively.
    fmt : str
        Numeric format string, e.g. ".2f".
    cmap : str or Colormap
        Matplotlib colormap for imshow.
    save_path : str or None
        If provided, figure is saved to this path (e.g. "out.png").
    """
    # Build the 4×2 data matrix
    mat = np.vstack([d.reshape(2,) for d in (data1, data2, data3, data4)])
    limbs = ['RF', 'LF', 'LH', 'RH']
    axes_labels = ['GRF_Y', 'GRF_Z']

    # Create figure + single Axes
    fig, ax = plt.subplots(figsize=(5, 6))

    # Show matrix as an image
    im = ax.imshow(mat, cmap=cmap, aspect=0.3)
    # Set ticks & labels
    ax.set_xticks(np.arange(len(axes_labels)))
    ax.set_yticks(np.arange(len(limbs)))
    ax.set_xticklabels(axes_labels)
    ax.set_yticklabels(limbs)

    # Remove the little tick “bars” by zeroing their length
    ax.tick_params(axis='both', which='both', length=0)

    # Annotate each cell with its value
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i,
                    format(mat[i, j], fmt),
                    ha="center", va="center",
                    color="black")

    # Optional colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.3, fraction=0.046, pad=0.08)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    cbar.locator = MaxNLocator(nbins=4)                        # :contentReference[oaicite:3]{index=3}
    cbar.update_ticks()       
    # cbar = fig.colorbar(im, ax=ax, shrink=1.0, pad=0.02)
    # Tight layout & save if requested
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    plt.show()



def plot_prediction_triangle(lf_y_predict, lf_z_predict, sigma=2, save_path=None):
    """
    Smooths and plots points from (lf_y_predict, lf_z_predict) and (lf_target_y_select, lf_target_z_select),
    and fills triangles formed by origin and consecutive points. Uses different colors for prediction and target.
    All axes, labels, and backgrounds are removed for a clean visualization.

    Parameters:
    - lf_y_predict: np.ndarray, shape (n,)
    - lf_z_predict: np.ndarray, shape (n,)
    - lf_target_z_select: np.ndarray, shape (n,)
    - lf_target_y_select: np.ndarray, shape (n,)
    - sigma: float - standard deviation for Gaussian kernel
    - save_path: str or None - if provided, saves the figure to this path
    """
    lf_y_predict = np.asarray(lf_y_predict).flatten()
    lf_z_predict = np.asarray(lf_z_predict).flatten()


    assert len(lf_y_predict) == len(lf_z_predict), "Y and Z arrays must have same length"


    # Smooth the data
    lf_y_predict_smooth = gaussian_filter1d(lf_y_predict, sigma=sigma)
    lf_z_predict_smooth = gaussian_filter1d(lf_z_predict, sigma=sigma)

    plt.figure(figsize=(8, 6))

    # Plot triangles for prediction (red)
    for i in range(1, len(lf_y_predict_smooth)):
        x_triangle = [0, lf_y_predict_smooth[i-1], lf_y_predict_smooth[i]]
        y_triangle = [0, lf_z_predict_smooth[i-1], lf_z_predict_smooth[i]]
        plt.fill(x_triangle, y_triangle, color=colors[4], alpha=0.5)


    # Plot points (no labels)
    plt.scatter(lf_y_predict_smooth, lf_z_predict_smooth, color=colors[4], s=4)


    # Remove all axes, ticks, labels, and background
    plt.axis('off')  # Turns off all axes and labels
    plt.gca().set_facecolor('none')  # Transparent background
    plt.grid(False)  # No grid
    plt.xlim([-3,3.7])
    plt.ylim([-3,3.7])
    # Ensure equal aspect ratio (keeps geometry correct)
    # plt.gca().set_aspect('equal')

    # Tight layout and save (if requested)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()



def plot_prediction_triangles(lf_y_predict, lf_z_predict, lf_target_y_select, lf_target_z_select, sigma=2, save_path=None):
    """
    Smooths and plots points from (lf_y_predict, lf_z_predict) and (lf_target_y_select, lf_target_z_select),
    and fills triangles formed by origin and consecutive points. Uses different colors for prediction and target.
    All axes, labels, and backgrounds are removed for a clean visualization.

    Parameters:
    - lf_y_predict: np.ndarray, shape (n,)
    - lf_z_predict: np.ndarray, shape (n,)
    - lf_target_z_select: np.ndarray, shape (n,)
    - lf_target_y_select: np.ndarray, shape (n,)
    - sigma: float - standard deviation for Gaussian kernel
    - save_path: str or None - if provided, saves the figure to this path
    """
    lf_y_predict = np.asarray(lf_y_predict).flatten()
    lf_z_predict = np.asarray(lf_z_predict).flatten()
    lf_target_y_select = np.asarray(lf_target_y_select).flatten()
    lf_target_z_select = np.asarray(lf_target_z_select).flatten()

    assert len(lf_y_predict) == len(lf_z_predict), "Y and Z arrays must have same length"
    assert len(lf_target_y_select) == len(lf_target_z_select), "Target Y and Z arrays must have same length"

    # Smooth the data
    lf_y_predict_smooth = gaussian_filter1d(lf_y_predict, sigma=sigma)
    lf_z_predict_smooth = gaussian_filter1d(lf_z_predict, sigma=sigma)
    lf_target_y_smooth = gaussian_filter1d(lf_target_y_select, sigma=sigma)
    lf_target_z_smooth = gaussian_filter1d(lf_target_z_select, sigma=sigma)

    plt.figure(figsize=(8, 6))

    # Plot triangles for prediction (red)
    for i in range(1, len(lf_y_predict_smooth)):
        x_triangle = [0, lf_y_predict_smooth[i-1], lf_y_predict_smooth[i]]
        y_triangle = [0, lf_z_predict_smooth[i-1], lf_z_predict_smooth[i]]
        plt.fill(x_triangle, y_triangle, color=colors[1], alpha=0.5)

    # Plot triangles for target (blue)
    for i in range(1, len(lf_target_y_smooth)):
        x_triangle = [0, lf_target_y_smooth[i-1], lf_target_y_smooth[i]]
        y_triangle = [0, lf_target_z_smooth[i-1], lf_target_z_smooth[i]]
        plt.fill(x_triangle, y_triangle, color=colors[4], alpha=0.3)

    # Plot points (no labels)
    plt.scatter(lf_y_predict_smooth, lf_z_predict_smooth, color=colors[1], s=4)
    plt.scatter(lf_target_y_smooth, lf_target_z_smooth, color=colors[4], s=4)

    # Remove all axes, ticks, labels, and background
    plt.axis('off')  # Turns off all axes and labels
    plt.gca().set_facecolor('none')  # Transparent background
    plt.grid(False)  # No grid
    plt.xlim([-3,3.7])
    plt.ylim([-3,3.7])
    # Ensure equal aspect ratio (keeps geometry correct)
    # plt.gca().set_aspect('equal')

    # Tight layout and save (if requested)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()