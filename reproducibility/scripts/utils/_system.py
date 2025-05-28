#!/usr/bin/env python3

import os
import platform
import re
import socket
import uuid
from datetime import datetime

import GPUtil
import psutil
import torch

# Try to import colorama for colorful output; if unavailable, define dummies.
try:
    from colorama import Fore, Style, init

    init(autoreset=True)
except ImportError:

    class DummyColor:
        """Dummy colors."""

        RESET = ""
        BRIGHT = ""
        CYAN = ""
        GREEN = ""
        YELLOW = ""
        MAGENTA = ""
        BLUE = ""
        RED = ""

    Fore = Style = DummyColor()


def get_size(num_bytes, suffix="B"):
    """Scale bytes to a proper format (e.g., 1253656 -> '1.20MB')."""
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if num_bytes < factor:
            return f"{num_bytes:.2f}{unit}{suffix}"
        num_bytes /= factor


def print_header(title, sep="="):
    """Print a formatted header with the given title."""
    line = sep * (len(title) + 12)
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{line}")
    print(f"     {title}")
    print(f"{line}{Style.RESET_ALL}")


def display_device_and_gpu_info():
    """Display overall device information.

    It includes Torch version, CUDA availability, and the primary device (CPU or GPU).
    """
    print_header("DEVICE INFORMATION", "-")
    torch_version = torch.__version__
    cuda_available = torch.cuda.is_available()
    cuda_version = torch.version.cuda if cuda_available else "N/A"
    print(f"{Fore.GREEN}Torch Version   :{Style.RESET_ALL} {torch_version}")
    print(f"{Fore.GREEN}CUDA Available  :{Style.RESET_ALL} {cuda_available}")
    print(f"{Fore.GREEN}CUDA Version    :{Style.RESET_ALL} {cuda_version}")

    device = torch.device("cuda" if cuda_available else "cpu")
    print(f"{Fore.GREEN}Primary Device  :{Style.RESET_ALL} {device}")


def display_gpu_info():
    """Display detailed GPU information using both Torch and GPUtil.

    This function shows for each GPU:
      - Torch device properties (name, compute capability, multiprocessor count, etc.)
      - A GPUtil summary (load, memory usage, temperature, and UUID).
    """
    print_header("GPU INFORMATION", "=")
    if torch.cuda.is_available():
        # Torch-based GPU details.
        num_gpus = torch.cuda.device_count()
        print(f"{Fore.GREEN}Number of GPUs detected (Torch):{Style.RESET_ALL} {num_gpus}")
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"\n{Fore.MAGENTA}GPU {i}:{Style.RESET_ALL} {gpu_name}")
            try:
                props = torch.cuda.get_device_properties(i)
                # Iterate over all properties provided by Torch.
                for field in props._fields:
                    value = getattr(props, field)
                    if field == "total_memory":
                        value = get_size(value)
                    print(f"  {Fore.YELLOW}{field.capitalize():20s}:{Style.RESET_ALL} {value}")
            except Exception as e:
                print(f"  {Fore.RED}Error retrieving torch GPU properties: {e}{Style.RESET_ALL}")

        # GPUtil-based GPU summary.
        gpus = GPUtil.getGPUs()
        if gpus:
            print_header("GPU SUMMARY (GPUtil)", "=")
            for idx, gpu in enumerate(gpus, start=1):
                print(f"\n{Fore.MAGENTA}GPU {idx}:{Style.RESET_ALL}")
                print(f"  {Fore.YELLOW}ID           :{Style.RESET_ALL} {gpu.id}")
                print(f"  {Fore.YELLOW}Name         :{Style.RESET_ALL} {gpu.name}")
                print(f"  {Fore.YELLOW}Load         :{Style.RESET_ALL} {gpu.load * 100:.2f}%")
                print(f"  {Fore.YELLOW}Free Memory  :{Style.RESET_ALL} {gpu.memoryFree} MB")
                print(f"  {Fore.YELLOW}Used Memory  :{Style.RESET_ALL} {gpu.memoryUsed} MB")
                print(f"  {Fore.YELLOW}Total Memory :{Style.RESET_ALL} {gpu.memoryTotal} MB")
                print(f"  {Fore.YELLOW}Temperature  :{Style.RESET_ALL} {gpu.temperature} °C")
                print(f"  {Fore.YELLOW}UUID         :{Style.RESET_ALL} {gpu.uuid}")
        else:
            print(f"{Fore.RED}No GPU details available via GPUtil.{Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}CUDA device not available. No GPU information to display.{Style.RESET_ALL}")


def display_system_info():
    """Display operating system and hardware details with additional context."""
    print_header("SYSTEM INFORMATION", "=")
    uname = platform.uname()
    print(f"{Fore.GREEN}System         :{Style.RESET_ALL} {uname.system}")
    print(f"{Fore.GREEN}Node Name      :{Style.RESET_ALL} {uname.node}")
    print(f"{Fore.GREEN}Release        :{Style.RESET_ALL} {uname.release}")
    print(f"{Fore.GREEN}Version        :{Style.RESET_ALL} {uname.version}")
    print(f"{Fore.GREEN}Machine        :{Style.RESET_ALL} {uname.machine}")
    print(f"{Fore.GREEN}Processor      :{Style.RESET_ALL} {uname.processor}")
    print(f"{Fore.GREEN}Platform       :{Style.RESET_ALL} {platform.platform()}")
    print(f"{Fore.GREEN}Architecture   :{Style.RESET_ALL} {platform.architecture()[0]}")
    print(f"{Fore.GREEN}Python Version :{Style.RESET_ALL} {platform.python_version()}")


def display_boot_time():
    """Display the system boot time and the current uptime in days, hours, minutes, and seconds."""
    print_header("BOOT TIME & UPTIME", "=")
    boot_time_timestamp = psutil.boot_time()
    boot_time = datetime.fromtimestamp(boot_time_timestamp)
    print(f"{Fore.GREEN}Boot Time :{Style.RESET_ALL} {boot_time.strftime('%Y/%m/%d %H:%M:%S')}")
    now = datetime.now()
    uptime = now - boot_time
    days, remainder = divmod(uptime.total_seconds(), 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(
        f"{Fore.GREEN}Uptime    :{Style.RESET_ALL} {int(days)} days, {int(hours)} hours, "
        f"{int(minutes)} minutes, {int(seconds)} seconds"
    )


def display_cpu_info():
    """Display detailed CPU information.

    This includes:
    - Core counts and frequency details
    - Overall CPU times and per-core usage percentages
    - CPU statistics (context switches, interrupts, etc.)
    - Load averages (if available)
    """
    print_header("CPU INFORMATION", "=")
    physical_cores = psutil.cpu_count(logical=False)
    total_cores = psutil.cpu_count(logical=True)
    print(f"{Fore.GREEN}Physical cores   :{Style.RESET_ALL} {physical_cores}")
    print(f"{Fore.GREEN}Total cores      :{Style.RESET_ALL} {total_cores}")

    cpufreq = psutil.cpu_freq()
    print(f"{Fore.GREEN}Max Frequency    :{Style.RESET_ALL} {cpufreq.max:.2f} MHz")
    print(f"{Fore.GREEN}Min Frequency    :{Style.RESET_ALL} {cpufreq.min:.2f} MHz")
    print(f"{Fore.GREEN}Current Frequency:{Style.RESET_ALL} {cpufreq.current:.2f} MHz")

    print_header("OVERALL CPU TIMES", "=")
    cpu_times = psutil.cpu_times()
    print(f"{Fore.GREEN}User    :{Style.RESET_ALL} {cpu_times.user:.2f} seconds")
    print(f"{Fore.GREEN}System  :{Style.RESET_ALL} {cpu_times.system:.2f} seconds")
    print(f"{Fore.GREEN}Idle    :{Style.RESET_ALL} {cpu_times.idle:.2f} seconds")
    if hasattr(cpu_times, "interrupt"):
        print(f"{Fore.GREEN}Interrupt:{Style.RESET_ALL} {cpu_times.interrupt:.2f} seconds")
    if hasattr(cpu_times, "dpc"):
        print(f"{Fore.GREEN}DPC     :{Style.RESET_ALL} {cpu_times.dpc:.2f} seconds")

    print_header("PER-CORE USAGE", "=")
    per_core_usage = psutil.cpu_percent(percpu=True, interval=0.5)
    for idx, usage in enumerate(per_core_usage):
        print(f"  {Fore.YELLOW}Core {idx:2d}:{Style.RESET_ALL} {usage:.2f}%")

    print_header("CPU STATISTICS", "=")
    cpu_stats = psutil.cpu_stats()
    print(f"{Fore.GREEN}Context Switches :{Style.RESET_ALL} {cpu_stats.ctx_switches}")
    print(f"{Fore.GREEN}Interrupts       :{Style.RESET_ALL} {cpu_stats.interrupts}")
    print(f"{Fore.GREEN}Soft Interrupts  :{Style.RESET_ALL} {cpu_stats.soft_interrupts}")
    print(f"{Fore.GREEN}System Calls     :{Style.RESET_ALL} {cpu_stats.syscalls}")

    try:
        load1, load5, load15 = os.getloadavg()
        print_header("LOAD AVERAGE (Unix)", "=")
        print(f"{Fore.GREEN}1 min  :{Style.RESET_ALL} {load1:.2f}")
        print(f"{Fore.GREEN}5 min  :{Style.RESET_ALL} {load5:.2f}")
        print(f"{Fore.GREEN}15 min :{Style.RESET_ALL} {load15:.2f}")
    except (AttributeError, OSError):
        print(f"{Fore.YELLOW}Load Average not supported on this platform.{Style.RESET_ALL}")


def display_memory_info():
    """Display RAM and swap memory usage details, including advanced virtual memory metrics."""
    print_header("MEMORY INFORMATION", "=")
    svmem = psutil.virtual_memory()
    print(f"{Fore.GREEN}Total     :{Style.RESET_ALL} {get_size(svmem.total)}")
    print(f"{Fore.GREEN}Available :{Style.RESET_ALL} {get_size(svmem.available)}")
    print(f"{Fore.GREEN}Used      :{Style.RESET_ALL} {get_size(svmem.used)}")
    print(f"{Fore.GREEN}Percentage:{Style.RESET_ALL} {svmem.percent}%")

    if hasattr(svmem, "active"):
        print(f"{Fore.GREEN}Active    :{Style.RESET_ALL} {get_size(svmem.active)}")
    if hasattr(svmem, "inactive"):
        print(f"{Fore.GREEN}Inactive  :{Style.RESET_ALL} {get_size(svmem.inactive)}")
    if hasattr(svmem, "buffers"):
        print(f"{Fore.GREEN}Buffers   :{Style.RESET_ALL} {get_size(svmem.buffers)}")
    if hasattr(svmem, "cached"):
        print(f"{Fore.GREEN}Cached    :{Style.RESET_ALL} {get_size(svmem.cached)}")
    if hasattr(svmem, "shared"):
        print(f"{Fore.GREEN}Shared    :{Style.RESET_ALL} {get_size(svmem.shared)}")
    if hasattr(svmem, "slab"):
        print(f"{Fore.GREEN}Slab      :{Style.RESET_ALL} {get_size(svmem.slab)}")

    print_header("SWAP MEMORY", "=")
    swap = psutil.swap_memory()
    print(f"{Fore.GREEN}Total     :{Style.RESET_ALL} {get_size(swap.total)}")
    print(f"{Fore.GREEN}Free      :{Style.RESET_ALL} {get_size(swap.free)}")
    print(f"{Fore.GREEN}Used      :{Style.RESET_ALL} {get_size(swap.used)}")
    print(f"{Fore.GREEN}Percentage:{Style.RESET_ALL} {swap.percent}%")


def display_disk_info():
    """Display disk partitions, usage statistics, and I/O data."""
    print_header("DISK INFORMATION", "=")
    print(f"{Fore.BLUE}Partitions and Usage:{Style.RESET_ALL}")
    partitions = psutil.disk_partitions()
    for partition in partitions:
        print(f"\n{Fore.MAGENTA}Device       :{Style.RESET_ALL} {partition.device}")
        print(f"  {Fore.YELLOW}Mountpoint :{Style.RESET_ALL} {partition.mountpoint}")
        print(f"  {Fore.YELLOW}File system:{Style.RESET_ALL} {partition.fstype}")
        try:
            partition_usage = psutil.disk_usage(partition.mountpoint)
        except PermissionError:
            print(f"  {Fore.RED}Permission Denied for partition usage.{Style.RESET_ALL}")
            continue
        print(f"  {Fore.YELLOW}Total Size :{Style.RESET_ALL} {get_size(partition_usage.total)}")
        print(f"  {Fore.YELLOW}Used       :{Style.RESET_ALL} {get_size(partition_usage.used)}")
        print(f"  {Fore.YELLOW}Free       :{Style.RESET_ALL} {get_size(partition_usage.free)}")
        print(f"  {Fore.YELLOW}Percentage :{Style.RESET_ALL} {partition_usage.percent}%")

    disk_io = psutil.disk_io_counters()
    print(f"\n{Fore.BLUE}I/O Statistics:{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Total read  :{Style.RESET_ALL} {get_size(disk_io.read_bytes)}")
    print(f"{Fore.GREEN}Total write :{Style.RESET_ALL} {get_size(disk_io.write_bytes)}")


def display_network_info():
    """Display network information including hostname, IP, MAC addresses, active interface details, and network I/O."""
    print_header("NETWORK INFORMATION", "=")
    hostname = socket.gethostname()
    try:
        ip_address = socket.gethostbyname(hostname)
    except socket.gaierror:
        ip_address = "N/A"
    mac_address = ":".join(re.findall("..", f"{uuid.getnode():012x}"))
    print(f"{Fore.GREEN}Hostname    :{Style.RESET_ALL} {hostname}")
    print(f"{Fore.GREEN}IP Address  :{Style.RESET_ALL} {ip_address}")
    print(f"{Fore.GREEN}MAC Address :{Style.RESET_ALL} {mac_address}")

    if_addrs = psutil.net_if_addrs()
    if_stats = psutil.net_if_stats()
    print_header("ACTIVE INTERFACE DETAILS", "-")
    for iface, stats in if_stats.items():
        if not stats.isup:
            continue  # Only display active interfaces
        addresses = if_addrs.get(iface, [])
        print(f"\n{Fore.MAGENTA}Interface: {iface}{Style.RESET_ALL}")
        for address in addresses:
            if address.family == socket.AF_INET:
                print(f"  {Fore.YELLOW}IP Address :{Style.RESET_ALL} {address.address}")
                print(f"  {Fore.YELLOW}Netmask    :{Style.RESET_ALL} {address.netmask}")
                print(f"  {Fore.YELLOW}Broadcast  :{Style.RESET_ALL} {address.broadcast}")
            elif address.family == socket.AF_INET6:
                print(f"  {Fore.YELLOW}IPv6 Address :{Style.RESET_ALL} {address.address}")
                print(f"  {Fore.YELLOW}Netmask      :{Style.RESET_ALL} {address.netmask}")
                print(f"  {Fore.YELLOW}Broadcast    :{Style.RESET_ALL} {address.broadcast}")
            else:
                print(f"  {Fore.YELLOW}MAC Address :{Style.RESET_ALL} {address.address}")

    net_io = psutil.net_io_counters()
    print_header("NETWORK I/O", "-")
    print(f"{Fore.GREEN}Total Bytes Sent    :{Style.RESET_ALL} {get_size(net_io.bytes_sent)}")
    print(f"{Fore.GREEN}Total Bytes Received:{Style.RESET_ALL} {get_size(net_io.bytes_recv)}")


def main():
    """Main function to execute all display routines."""
    print(f"\n{Fore.BLUE}{Style.BRIGHT}{'#' * 10} Detailed System Information {'#' * 10}{Style.RESET_ALL}\n")
    display_device_and_gpu_info()
    display_gpu_info()
    display_system_info()
    display_boot_time()
    display_cpu_info()
    display_memory_info()
    display_disk_info()
    display_network_info()
    print(f"\n{Fore.BLUE}{Style.BRIGHT}{'#' * 10} End of Information {'#' * 10}{Style.RESET_ALL}\n")


if __name__ == "__main__":
    main()
