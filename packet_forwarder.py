import serial
import struct
import socket
import time
import mmap
import struct


# --- config ---
DEVICE       = 'COM3'
BAUDRATE     = 115200
PACKET_LEN   = 518  # 6 bytes MAC + 512 bytes float32
MAC_HEADER   = b'\x00\x80\xE1\x12\x34\x56'
MAX_BUFFER   = 2048

RAW_IFACE    = 'enp1s0'  # Replace with your actual interface name
DEST_MAC     = b'\xFF\xFF\xFF\xFF\xFF\xFF'       # Broadcast
SRC_MAC      = MAC_HEADER
ETH_TYPE     = b'\x12\x34'                       # Custom EtherType

# --- serial port ---
ser = serial.Serial(DEVICE, BAUDRATE, timeout=1)

# --- raw Ethernet socket ---
#sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW)
#sock.bind((RAW_IFACE, 0))

def sync_to_header(buffer):
    """Find start of packet by MAC header match"""
    for i in range(len(buffer) - len(MAC_HEADER)):
        if buffer[i:i+6] == MAC_HEADER:
            return i
    return -1

def read_packet():
    """Read and return one complete packet from serial"""
    buffer = bytearray()
    while True:
        buffer += ser.read(MAX_BUFFER)
        offset = sync_to_header(buffer)
        if offset >= 0 and len(buffer) - offset >= PACKET_LEN:
            packet = buffer[offset:offset+PACKET_LEN]
            return packet
        buffer = buffer[-PACKET_LEN:]  # Retain last chunk

def parse_packet(packet):
    """Unpack MAC and voltage row from packet"""
    mac = packet[0:6]
    volt_row = struct.unpack('<128f', packet[6:])
    return mac, volt_row

# --- main loop ---
TOTAL_SAMPLES = 9999872
FLOATS_PER_PACKET = 128
FLOAT_SIZE = 4
PACKET_SIZE = FLOATS_PER_PACKET * FLOAT_SIZE

MMAP_SIZE = 16 * 1024 * 1024  # 16MB
LINES_PER_FILE = MMAP_SIZE // PACKET_SIZE
TOTAL_FILES = 32

sample_count = 0
file_index = 0
line_index = 0

def create_mmap_file(index):
    filename = f"mmfile{index + 1}"
    with open(filename, 'wb') as f:
        f.write(b'\x00' * MMAP_SIZE)
    f = open(filename, 'r+b')
    return f, mmap.mmap(f.fileno(), 0)

mm_file, mm_map = create_mmap_file(file_index)

while sample_count < TOTAL_SAMPLES:
    raw = read_packet()
    _, voltages = parse_packet(raw)
    payload = struct.pack('<128f', *voltages)

    # Write to mmap
    offset = line_index * PACKET_SIZE
    mm_map[offset:offset + PACKET_SIZE] = payload

    # Optional console echo (still helpful!)
    line = " ".join(f"{v:.4f}" for v in voltages)
    print(line)

    sample_count += 1
    line_index += 1

    if line_index >= LINES_PER_FILE:
        mm_map.close()
        mm_file.close()
        file_index += 1
        line_index = 0
        if file_index >= TOTAL_FILES:
            break
        mm_file, mm_map = create_mmap_file(file_index)

mm_map.close()
mm_file.close()



