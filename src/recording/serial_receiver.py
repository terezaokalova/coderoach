import serial
import matplotlib.pyplot as plt
from collections import deque

PORT = "/dev/cu.usbmodem1101"
BAUD = 115200

ser = serial.Serial(PORT, BAUD, timeout=1)

samples = deque(maxlen=500)

plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([])
ax.set_title("GSR signal")
ax.set_xlabel("Sample")
ax.set_ylabel("ADC reading")
ax.set_ylim(0, 4095)

try:
    while True:
        message = ser.readline().decode("utf-8", errors="ignore").strip()

        # Arduino sends lines such as: GSR:2037
        if message.startswith("GSR:"):
            try:
                value = float(message.split(":", 1)[1])
            except ValueError:
                continue

            samples.append(value)
            print(value)

            line.set_data(range(len(samples)), samples)
            ax.set_xlim(0, max(100, len(samples)))
            plt.pause(0.001)

except KeyboardInterrupt:
    print("\nStopped")

finally:
    ser.close()