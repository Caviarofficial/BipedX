"""
信号滤波程序：结合卡尔曼滤波与互补滤波
Signal Filtering Program: Kalman Filter & Complementary Filter Combined
Author: AI Assistant
功能：输入带噪声的正弦波数据，输出滤波后的平滑曲线，并展示对比图像
Function: Input noisy sine wave data, output filtered smooth curve, and display comparison graphs
"""


# 检查并安装必要的库
# Check and install necessary libraries
def check_and_install_packages():
    """检查必要的包是否已安装，如果未安装则提示安装"""
    """Check if required packages are installed, prompt for installation if not"""
    required_packages = ['numpy', 'matplotlib', 'scipy']
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("=" * 60)
        print("缺少必要的Python库，请安装以下包：")
        print("Missing required Python libraries. Please install:")
        print(" ".join(missing_packages))
        print("\n安装方法：")
        print("\nInstallation methods:")
        print("1. 在PyCharm的Terminal中运行：")
        print("1. Run in PyCharm Terminal:")
        print(f"   pip install {' '.join(missing_packages)}")
        print("\n2. 或者使用清华镜像源（国内推荐）：")
        print("\n2. Or use Tsinghua Mirror (recommended for China):")
        print(f"   pip install {' '.join(missing_packages)} -i https://pypi.tuna.tsinghua.edu.cn/simple")
        print("=" * 60)
        return False
    return True


# 如果检查通过，导入库
# If check passes, import libraries
if check_and_install_packages():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import signal

    print("所有必要的库都已安装，开始运行滤波程序...")
    print("All required libraries are installed. Starting filter program...")
else:
    print("请先安装必要的库，然后重新运行程序。")
    print("Please install required libraries first, then rerun the program.")
    exit(1)


class SimpleKalmanFilter:
    """简化的卡尔曼滤波器（一维版本，更易理解）"""
    """Simplified Kalman Filter (1D version, easier to understand)"""

    def __init__(self, process_noise=1e-5, measurement_noise=1e-1):
        """
        初始化卡尔曼滤波器
        Initialize Kalman Filter
        :param process_noise: 过程噪声（越小越信任预测）
        :param process_noise: Process noise (smaller values trust predictions more)
        :param measurement_noise: 测量噪声（越小越信任测量）
        :param measurement_noise: Measurement noise (smaller values trust measurements more)
        """
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        # 初始估计值
        # Initial estimate
        self.estimated_value = 0.0

        # 初始估计误差
        # Initial estimation error
        self.estimation_error = 1.0

        # 预测误差
        # Prediction error
        self.prediction_error = 0.0

        # 卡尔曼增益
        # Kalman gain
        self.kalman_gain = 0.0

    def update(self, measurement):
        """
        更新卡尔曼滤波器状态
        Update Kalman Filter state
        :param measurement: 测量值
        :param measurement: Measurement value
        :return: 滤波后的值
        :return: Filtered value
        """
        # 预测步骤
        # Prediction step
        self.prediction_error = self.estimation_error + self.process_noise

        # 计算卡尔曼增益
        # Calculate Kalman gain
        self.kalman_gain = self.prediction_error / (self.prediction_error + self.measurement_noise)

        # 更新估计值
        # Update estimate
        self.estimated_value = self.estimated_value + self.kalman_gain * (measurement - self.estimated_value)

        # 更新估计误差
        # Update estimation error
        self.estimation_error = (1 - self.kalman_gain) * self.prediction_error

        return self.estimated_value


class ComplementaryFilter:
    """互补滤波器"""
    """Complementary Filter"""

    def __init__(self, alpha=0.95):
        """
        初始化互补滤波器
        Initialize Complementary Filter
        :param alpha: 滤波系数 (0-1之间，越大表示对历史值信任度越高)
        :param alpha: Filter coefficient (0-1, higher values trust historical values more)
        """
        self.alpha = alpha
        self.filtered_value = 0.0
        self.prev_value = 0.0

    def update(self, measurement, dt=0.01):
        """
        更新互补滤波器
        Update Complementary Filter
        :param measurement: 测量值
        :param measurement: Measurement value
        :param dt: 时间间隔（秒）
        :param dt: Time interval (seconds)
        :return: 滤波后的值
        :return: Filtered value
        """
        # 简单的互补滤波公式
        # Simple complementary filter formula
        # 当前值 = α × 前一时刻滤波值 + (1-α) × 当前测量值
        # Current value = α × previous filtered value + (1-α) × current measurement
        self.filtered_value = self.alpha * self.filtered_value + (1 - self.alpha) * measurement

        return self.filtered_value


class HybridFilter:
    """混合滤波器：结合卡尔曼滤波和互补滤波"""
    """Hybrid Filter: Combines Kalman and Complementary Filters"""

    def __init__(self, kf_alpha=0.7, cf_alpha=0.95):
        """
        初始化混合滤波器
        Initialize Hybrid Filter
        :param kf_alpha: 卡尔曼滤波权重
        :param kf_alpha: Kalman filter weight
        :param cf_alpha: 互补滤波系数
        :param cf_alpha: Complementary filter coefficient
        """
        self.kalman_filter = SimpleKalmanFilter()
        self.complementary_filter = ComplementaryFilter(cf_alpha)
        self.kf_weight = kf_alpha
        self.cf_weight = 1.0 - kf_alpha

    def update(self, measurement):
        """
        更新混合滤波器
        Update Hybrid Filter
        :param measurement: 测量值
        :param measurement: Measurement value
        :return: 混合滤波后的值
        :return: Hybrid filtered value
        """
        # 卡尔曼滤波
        # Kalman filter
        kalman_value = self.kalman_filter.update(measurement)

        # 互补滤波
        # Complementary filter
        complementary_value = self.complementary_filter.update(measurement)

        # 加权结合两种滤波结果
        # Weighted combination of both filter results
        hybrid_value = self.kf_weight * kalman_value + self.cf_weight * complementary_value

        return hybrid_value


def generate_sine_wave_with_noise():
    """
    生成带噪声的正弦波数据
    Generate noisy sine wave data
    :return: 时间序列，纯净正弦波，带噪声的正弦波
    :return: Time series, pure sine wave, noisy sine wave
    """
    # 参数设置
    # Parameter settings
    duration = 5.0  # 信号时长（秒）Signal duration (seconds)
    sampling_rate = 200  # 采样率（Hz）Sampling rate (Hz)
    frequency = 2.0  # 正弦波频率（Hz）Sine wave frequency (Hz)
    amplitude = 1.0  # 正弦波振幅 Sine wave amplitude

    # 生成时间序列
    # Generate time series
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

    # 生成纯净正弦波
    # Generate pure sine wave
    pure_signal = amplitude * np.sin(2 * np.pi * frequency * t)

    # 生成噪声
    # Generate noise
    # 1. 高斯白噪声
    # 1. Gaussian white noise
    gaussian_noise = 0.3 * np.random.randn(len(t))

    # 2. 脉冲噪声（偶尔的大幅度噪声）
    # 2. Impulse noise (occasional large amplitude noise)
    impulse_noise = np.zeros(len(t))
    impulse_indices = np.random.choice(len(t), size=int(0.02 * len(t)), replace=False)
    impulse_noise[impulse_indices] = 1.5 * np.random.randn(len(impulse_indices))

    # 3. 低频漂移噪声
    # 3. Low-frequency drift noise
    drift_noise = 0.1 * np.sin(2 * np.pi * 0.1 * t)

    # 合成带噪声的信号
    # Combine into noisy signal
    noisy_signal = pure_signal + gaussian_noise + impulse_noise + drift_noise

    return t, pure_signal, noisy_signal


def apply_moving_average_filter(signal_data, window_size=5):
    """
    应用移动平均滤波器
    Apply moving average filter
    :param signal_data: 输入信号
    :param signal_data: Input signal
    :param window_size: 窗口大小
    :param window_size: Window size
    :return: 滤波后的信号
    :return: Filtered signal
    """
    # 使用卷积实现移动平均
    # Implement moving average using convolution
    window = np.ones(window_size) / window_size
    filtered = np.convolve(signal_data, window, mode='same')
    return filtered


def calculate_performance_metrics(original, filtered):
    """
    计算滤波性能指标
    Calculate filtering performance metrics
    :param original: 原始纯净信号
    :param original: Original pure signal
    :param filtered: 滤波后的信号
    :param filtered: Filtered signal
    :return: 性能指标字典
    :return: Performance metrics dictionary
    """
    # 均方根误差 (RMSE)
    # Root Mean Square Error (RMSE)
    rmse = np.sqrt(np.mean((original - filtered) ** 2))

    # 平均绝对误差 (MAE)
    # Mean Absolute Error (MAE)
    mae = np.mean(np.abs(original - filtered))

    # 峰值信噪比 (PSNR)
    # Peak Signal-to-Noise Ratio (PSNR)
    max_val = np.max(original)
    mse = np.mean((original - filtered) ** 2)
    psnr = 10 * np.log10(max_val ** 2 / mse) if mse > 0 else float('inf')

    # 信号平滑度（通过标准差衡量）
    # Signal smoothness (measured by standard deviation of differences)
    smoothness = np.std(np.diff(filtered))

    return {
        'RMSE': rmse,
        'MAE': mae,
        'PSNR': psnr,
        'Smoothness': smoothness
    }


def plot_filter_results(t, pure_signal, noisy_signal, filtered_signals, metrics):
    """
    绘制滤波结果图
    Plot filtering results
    :param t: 时间序列
    :param t: Time series
    :param pure_signal: 纯净信号
    :param pure_signal: Pure signal
    :param noisy_signal: 噪声信号
    :param noisy_signal: Noisy signal
    :param filtered_signals: 滤波后的信号字典
    :param filtered_signals: Dictionary of filtered signals
    :param metrics: 性能指标字典
    :param metrics: Performance metrics dictionary
    """
    # 创建图形
    # Create figure
    plt.figure(figsize=(15, 10))

    # 1. 原始信号与噪声信号对比
    # 1. Original signal vs noisy signal comparison
    plt.subplot(2, 3, 1)
    plt.plot(t, pure_signal, 'b-', linewidth=2, label='Pure Signal', alpha=0.8)
    plt.plot(t, noisy_signal, 'r-', linewidth=1, label='Noisy Signal', alpha=0.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Original Signal vs Noisy Signal')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. 卡尔曼滤波结果
    # 2. Kalman filter results
    plt.subplot(2, 3, 2)
    plt.plot(t, pure_signal, 'b-', linewidth=2, label='Pure Signal', alpha=0.8)
    plt.plot(t, filtered_signals['Kalman'], 'g-', linewidth=2, label='Kalman Filter', alpha=0.8)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'Kalman Filter\nRMSE: {metrics["Kalman"]["RMSE"]:.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3. 互补滤波结果
    # 3. Complementary filter results
    plt.subplot(2, 3, 3)
    plt.plot(t, pure_signal, 'b-', linewidth=2, label='Pure Signal', alpha=0.8)
    plt.plot(t, filtered_signals['Complementary'], 'm-', linewidth=2, label='Complementary Filter', alpha=0.8)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'Complementary Filter\nRMSE: {metrics["Complementary"]["RMSE"]:.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 4. 混合滤波结果
    # 4. Hybrid filter results
    plt.subplot(2, 3, 4)
    plt.plot(t, pure_signal, 'b-', linewidth=2, label='Pure Signal', alpha=0.8)
    plt.plot(t, filtered_signals['Hybrid'], 'orange', linewidth=2, label='Hybrid Filter', alpha=0.8)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'Hybrid Filter\nRMSE: {metrics["Hybrid"]["RMSE"]:.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 5. 移动平均滤波结果
    # 5. Moving average filter results
    plt.subplot(2, 3, 5)
    plt.plot(t, pure_signal, 'b-', linewidth=2, label='Pure Signal', alpha=0.8)
    plt.plot(t, filtered_signals['Moving Average'], 'purple', linewidth=2, label='Moving Average', alpha=0.8)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'Moving Average Filter\nRMSE: {metrics["Moving Average"]["RMSE"]:.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 6. 所有滤波方法对比
    # 6. All filter methods comparison
    plt.subplot(2, 3, 6)
    plt.plot(t, pure_signal, 'k--', linewidth=2, label='Pure Signal', alpha=0.8)
    plt.plot(t, filtered_signals['Kalman'], 'g-', linewidth=1, label='Kalman Filter', alpha=0.7)
    plt.plot(t, filtered_signals['Complementary'], 'm-', linewidth=1, label='Complementary Filter', alpha=0.7)
    plt.plot(t, filtered_signals['Hybrid'], 'orange', linewidth=2, label='Hybrid Filter', alpha=0.9)
    plt.plot(t, filtered_signals['Moving Average'], 'purple', linewidth=1, label='Moving Average', alpha=0.7)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('All Filter Methods Comparison')
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_error_comparison(t, pure_signal, filtered_signals):
    """
    绘制误差对比图
    Plot error comparison
    :param t: 时间序列
    :param t: Time series
    :param pure_signal: 纯净信号
    :param pure_signal: Pure signal
    :param filtered_signals: 滤波后的信号字典
    :param filtered_signals: Dictionary of filtered signals
    """
    plt.figure(figsize=(12, 8))

    colors = {'Kalman': 'green', 'Complementary': 'magenta',
              'Hybrid': 'orange', 'Moving Average': 'purple'}

    for i, (name, filtered) in enumerate(filtered_signals.items()):
        error = pure_signal - filtered

        plt.subplot(2, 2, i + 1)
        plt.plot(t, error, color=colors[name], linewidth=1.5, alpha=0.8)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.xlabel('Time (s)')
        plt.ylabel('Error')
        plt.title(f'{name} Filter Error')
        plt.grid(True, alpha=0.3)

        # 显示误差统计
        # Display error statistics
        mean_error = np.mean(error)
        std_error = np.std(error)
        plt.text(0.05, 0.95, f'Mean: {mean_error:.4f}\nStd Dev: {std_error:.4f}',
                 transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.show()


def main():
    """主函数"""
    """Main function"""
    print("=" * 60)
    print("信号滤波程序：卡尔曼滤波与互补滤波结合")
    print("Signal Filtering Program with Kalman and Complementary Filters")
    print("=" * 60)

    # 1. 生成带噪声的正弦波
    # 1. Generate noisy sine wave
    print("正在生成带噪声的正弦波信号...")
    print("Generating noisy sine wave signal...")
    t, pure_signal, noisy_signal = generate_sine_wave_with_noise()
    print(f"✓ 信号生成完成！共{len(t)}个采样点，采样率{len(t) / 5}Hz")
    print(f"✓ Signal generation complete! {len(t)} samples, sampling rate {len(t) / 5} Hz")

    # 2. 初始化各种滤波器
    # 2. Initialize various filters
    print("正在初始化滤波器...")
    print("Initializing filters...")
    kalman_filter = SimpleKalmanFilter(process_noise=1e-5, measurement_noise=0.1)
    complementary_filter = ComplementaryFilter(alpha=0.95)
    hybrid_filter = HybridFilter(kf_alpha=0.7, cf_alpha=0.95)

    # 3. 应用各种滤波器
    # 3. Apply various filters
    print("正在应用滤波器处理信号...")
    print("Applying filters to process signal...")

    # 存储各种滤波结果
    # Store various filtering results
    kalman_result = []
    complementary_result = []
    hybrid_result = []

    for measurement in noisy_signal:
        kalman_result.append(kalman_filter.update(measurement))
        complementary_result.append(complementary_filter.update(measurement))
        hybrid_result.append(hybrid_filter.update(measurement))

    # 移动平均滤波
    # Moving average filter
    moving_avg_result = apply_moving_average_filter(noisy_signal, window_size=7)

    # 整理结果
    # Organize results
    filtered_signals = {
        'Kalman': np.array(kalman_result),
        'Complementary': np.array(complementary_result),
        'Hybrid': np.array(hybrid_result),
        'Moving Average': moving_avg_result
    }

    # 4. 计算性能指标
    # 4. Calculate performance metrics
    print("正在计算滤波性能指标...")
    print("Calculating filtering performance metrics...")
    metrics = {}
    for name, filtered in filtered_signals.items():
        metrics[name] = calculate_performance_metrics(pure_signal, filtered)

    # 5. 显示性能指标
    # 5. Display performance metrics
    print("\n" + "=" * 60)
    print("滤波性能指标对比：")
    print("Filter Performance Metrics Comparison:")
    print("=" * 60)
    print(f"{'滤波器':<15} {'RMSE':<10} {'MAE':<10} {'PSNR(dB)':<12} {'平滑度':<10}")
    print(f"{'Filter':<15} {'RMSE':<10} {'MAE':<10} {'PSNR(dB)':<12} {'Smoothness':<10}")
    print("-" * 60)

    for name, metric in metrics.items():
        print(f"{name:<15} {metric['RMSE']:<10.4f} {metric['MAE']:<10.4f} "
              f"{metric['PSNR']:<12.2f} {metric['Smoothness']:<10.4f}")

    # 找到最佳滤波器
    # Find best filter
    best_filter = min(metrics.items(), key=lambda x: x[1]['RMSE'])
    print(f"\n✓ 最佳滤波器: {best_filter[0]} (RMSE: {best_filter[1]['RMSE']:.4f})")
    print(f"✓ Best filter: {best_filter[0]} (RMSE: {best_filter[1]['RMSE']:.4f})")

    # 6. 绘制结果
    # 6. Plot results
    print("\n正在绘制滤波结果图...")
    print("\nPlotting filtering results...")
    plot_filter_results(t, pure_signal, noisy_signal, filtered_signals, metrics)

    # 7. 绘制误差对比图
    # 7. Plot error comparison
    print("正在绘制误差对比图...")
    print("Plotting error comparison...")
    plot_error_comparison(t, pure_signal, filtered_signals)

    # 8. 保存结果
    # 8. Save results
    print("\n正在保存滤波结果到文件...")
    print("\nSaving filtering results to files...")

    # 保存数据到CSV文件
    # Save data to CSV file
    save_data = np.column_stack((
        t,
        pure_signal,
        noisy_signal,
        filtered_signals['Kalman'],
        filtered_signals['Complementary'],
        filtered_signals['Hybrid'],
        filtered_signals['Moving Average']
    ))

    np.savetxt('filtered_signal_results.csv', save_data, delimiter=',',
               header='Time,Pure Signal,Noisy Signal,Kalman Filter,Complementary Filter,Hybrid Filter,Moving Average Filter',
               fmt='%.6f', comments='')

    print("✓ 结果已保存到 'filtered_signal_results.csv'")
    print("✓ Results saved to 'filtered_signal_results.csv'")

    # 9. 显示总结
    # 9. Display summary
    print("\n" + "=" * 60)
    print("程序执行完成！总结：")
    print("Program execution complete! Summary:")
    print("=" * 60)
    print("1. 生成了带噪声的正弦波信号（包含高斯噪声、脉冲噪声和低频漂移）")
    print("1. Generated noisy sine wave signal (including Gaussian noise, impulse noise, and low-frequency drift)")
    print("2. 应用了四种滤波方法：")
    print("2. Applied four filtering methods:")
    print("   - 卡尔曼滤波：基于动态模型的最优估计")
    print("   - Kalman Filter: Optimal estimation based on dynamic model")
    print("   - 互补滤波：结合高频和低频信号的优势")
    print("   - Complementary Filter: Combining advantages of high and low frequency signals")
    print("   - 混合滤波：卡尔曼滤波与互补滤波的加权结合")
    print("   - Hybrid Filter: Weighted combination of Kalman and Complementary filters")
    print("   - 移动平均滤波：简单的滑动窗口平均")
    print("   - Moving Average Filter: Simple sliding window average")
    print("3. 计算了各项性能指标（RMSE、MAE、PSNR、平滑度）")
    print("3. Calculated performance metrics (RMSE, MAE, PSNR, Smoothness)")
    print("4. 绘制了滤波前后的对比图和误差分析图")
    print("4. Plotted comparison graphs before and after filtering, and error analysis graphs")
    print("5. 结果数据已保存到CSV文件")
    print("5. Result data saved to CSV file")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序被用户中断。")
        print("\nProgram interrupted by user.")
    except Exception as e:
        print(f"\n程序执行过程中出现错误: {e}")
        print(f"\nError occurred during program execution: {e}")
        print("请检查是否所有必要的库都已正确安装。")
        print("Please check if all required libraries are properly installed.")