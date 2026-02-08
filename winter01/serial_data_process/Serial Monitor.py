# simple_serial_monitor.py
import csv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
import os
from collections import deque
import random
import math


class SimpleSerialMonitor:
    def __init__(self, csv_file, update_interval=100, window_size=100):
        self.csv_file = csv_file
        self.update_interval = update_interval
        self.window_size = window_size

        # Read data
        self.headers, self.data = self.read_csv()

        if not self.data:
            print("Error: No data loaded!")
            return

        # Setup plot
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.lines = {}
        self.queues = {}
        self.time_queue = deque(maxlen=window_size)
        self.current_index = 0

        self.setup_plot()

    def read_csv(self):
        """Read CSV file"""
        data = []
        headers = []
        try:
            with open(self.csv_file, 'r') as f:
                reader = csv.reader(f)
                headers = next(reader)
                print(f"Columns: {headers}")

                for row in reader:
                    data.append([float(x) for x in row])

            print(f"Loaded {len(data)} rows")
            return headers, data
        except Exception as e:
            print(f"Error reading file: {e}")
            return [], []

    def setup_plot(self):
        """Setup plot with English labels"""
        self.ax.set_title('Serial Data Monitor', fontsize=14)
        self.ax.set_xlabel(self.headers[0] if self.headers else 'Index', fontsize=12)
        self.ax.set_ylabel('Value', fontsize=12)
        self.ax.grid(True, alpha=0.3)

        # Create lines for each data column (skip time column)
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        for i in range(1, min(len(self.headers), 7)):  # Limit to 6 data columns
            col_name = self.headers[i]
            self.queues[i] = deque(maxlen=self.window_size)

            color_idx = (i - 1) % len(colors)
            line, = self.ax.plot([], [],
                                 label=col_name,
                                 color=colors[color_idx],
                                 linewidth=2)
            self.lines[i] = line

        self.ax.legend()
        self.fig.tight_layout()

    def init_animation(self):
        """Initialize animation"""
        for line in self.lines.values():
            line.set_data([], [])
        return list(self.lines.values())

    def update_plot(self, frame):
        """Update plot"""
        if self.current_index >= len(self.data):
            return list(self.lines.values())

        # Get current data point
        current_row = self.data[self.current_index]

        # Update time queue
        self.time_queue.append(current_row[0])

        # Update data queues
        for i in self.lines.keys():
            if i < len(current_row):
                self.queues[i].append(current_row[i])
                self.lines[i].set_data(self.time_queue, self.queues[i])

        # Auto adjust axes
        if len(self.time_queue) > 0:
            x_min = min(self.time_queue)
            x_max = max(self.time_queue)
            x_range = x_max - x_min if x_max > x_min else 1

            all_y = []
            for queue in self.queues.values():
                all_y.extend(queue)

            if all_y:
                y_min = min(all_y)
                y_max = max(all_y)
                y_range = y_max - y_min if y_max > y_min else 1

                self.ax.set_xlim(x_min - 0.05 * x_range, x_max + 0.05 * x_range)
                self.ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

        # Update title
        self.ax.set_title(f'Serial Data - Point {self.current_index + 1}/{len(self.data)}')

        self.current_index += 1
        return list(self.lines.values())

    def start_animation(self):
        """Start animation"""
        if not self.data:
            print("No data to display!")
            return

        print(f"Starting animation...")
        print(f"Total points: {len(self.data)}")
        print(f"Update interval: {self.update_interval}ms")
        print(f"Window size: {self.window_size}")

        frames = len(self.data) + 10

        anim = FuncAnimation(self.fig,
                             self.update_plot,
                             init_func=self.init_animation,
                             frames=frames,
                             interval=self.update_interval,
                             blit=True,
                             repeat=False)

        plt.show()


def create_sample_data():
    """Create sample CSV data"""
    print("Creating sample data...")

    with open('sample_serial.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Time', 'Sensor1', 'Sensor2', 'Sensor3'])

        for i in range(100):
            t = i * 0.1
            s1 = 50 + 30 * math.sin(t) + random.uniform(-2, 2)
            s2 = 70 + 20 * math.cos(t * 0.5) + random.uniform(-3, 3)
            s3 = 30 + 10 * math.sin(t * 2) + random.uniform(-1, 1)
            writer.writerow([round(t, 2), round(s1, 2), round(s2, 2), round(s3, 2)])

    print("Created sample_serial.csv")


def main():
    parser = argparse.ArgumentParser(description='Simple Serial Data Monitor')
    parser.add_argument('--csv', type=str, help='CSV file path')
    parser.add_argument('--interval', type=int, default=100, help='Update interval (ms)')
    parser.add_argument('--window', type=int, default=50, help='Window size')
    parser.add_argument('--create-sample', action='store_true', help='Create sample data')

    args = parser.parse_args()

    if args.create_sample:
        create_sample_data()
        return

    csv_file = args.csv if args.csv else 'sample_serial.csv'

    if not os.path.exists(csv_file):
        print(f"File {csv_file} not found!")
        print("Creating sample data...")
        create_sample_data()

    monitor = SimpleSerialMonitor(
        csv_file=csv_file,
        update_interval=args.interval,
        window_size=args.window
    )

    if monitor.data:
        monitor.start_animation()


if __name__ == "__main__":
    main()