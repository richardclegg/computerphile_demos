#!/usr/bin/env python3
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from scapy.all import PcapReader, IP

INTERVAL = 0.1  # seconds per bin
GROUP    = 10   # bins per averaged point


def process_pcap(filename: Path, ip_filter: str):
    start_time = abs_start = None
    byte_count = 0
    times, bytes_mb = [], []
    avg_times, avg_mb = [], []
    group_ct, group_acc = 0, 0.0

    with PcapReader(str(filename)) as reader:
        for pkt in reader:
            if not pkt.haslayer(IP):
                continue
            ip = pkt[IP]
            if ip.src == ip_filter or ip.dst == ip_filter:
                byte_count += len(pkt)

            now = float(pkt.time)
            if start_time is None:
                start_time = abs_start = now

            if now - start_time >= INTERVAL:
                mb = byte_count / 1e6
                elapsed = now - abs_start
                times.append(elapsed)
                bytes_mb.append(mb)
                group_acc += mb
                group_ct  += 1
                if group_ct == GROUP:
                    avg_times.append(elapsed)
                    avg_mb.append(group_acc / GROUP)
                    group_acc = group_ct = 0
                print(f"{elapsed:.2f}s: {byte_count} bytes")
                byte_count = 0
                start_time = now

    return times, bytes_mb, avg_times, avg_mb


def main():
    if len(sys.argv) != 3:
        sys.exit(f"Usage: {sys.argv[0]} <ip-address> <pcap-file>")

    ip_filter, filename = sys.argv[1], Path(sys.argv[2])
    if not filename.exists():
        sys.exit(f"Error: '{filename}' not found.")

    times, bytes_mb, avg_times, avg_mb = process_pcap(filename, ip_filter)

    plt.plot(times,     bytes_mb, marker="o", label="per bin")
    plt.plot(avg_times, avg_mb,   marker="+", label=f"{GROUP}-bin average")
    plt.xlabel("Time (s)")
    plt.ylabel(f"MB/{int(INTERVAL*1000)} ms")
    plt.title("Throughput over time")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
