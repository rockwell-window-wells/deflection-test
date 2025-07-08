#!/usr/bin/env python3
"""
Enhanced pymodbus test script for ClickPlus PLC
Connects via TCP/Ethernet with comprehensive diagnostics
"""

from pymodbus.client import ModbusTcpClient
import socket
import time

# Connection parameters
PLC_IP = "192.168.0.10"
PLC_PORT = 502  # Standard Modbus TCP port
UNIT_ID = 4     # Modbus unit ID (slave ID)
ADDRESS = 28971 # Register address to read
TIMEOUT = 5     # Connection timeout in seconds

def test_network_connectivity():
    """Test basic network connectivity to PLC"""
    print("Testing network connectivity...")
    
    # Test ping (ICMP)
    import subprocess
    try:
        result = subprocess.run(['ping', '-c', '1', '-W', '2', PLC_IP], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"? Ping to {PLC_IP} successful")
        else:
            print(f"? Ping to {PLC_IP} failed")
            return False
    except Exception as e:
        print(f"? Ping test error: {e}")
        return False
    
    # Test TCP port connectivity
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        result = sock.connect_ex((PLC_IP, PLC_PORT))
        sock.close()
        
        if result == 0:
            print(f"? TCP port {PLC_PORT} is open on {PLC_IP}")
            return True
        else:
            print(f"? TCP port {PLC_PORT} is closed or filtered on {PLC_IP}")
            return False
    except Exception as e:
        print(f"? TCP port test error: {e}")
        return False

def test_multiple_addresses():
    """Test reading from multiple common addresses"""
    client = ModbusTcpClient(host=PLC_IP, port=PLC_PORT, timeout=TIMEOUT)
    
    # Common test addresses for ClickPlus PLCs
    test_addresses = [1, 100, 1000, 40001, 400001, ADDRESS]
    
    try:
        if not client.connect():
            print("Failed to connect for address testing")
            return
        
        print("\nTesting multiple addresses...")
        for addr in test_addresses:
            try:
                result = client.read_holding_registers(address=addr, count=1, unit=UNIT_ID)
                if not result.isError():
                    value = result.registers[0]
                    print(f"? Address {addr}: {value} (0x{value:04X})")
                else:
                    print(f"? Address {addr}: {result}")
            except Exception as e:
                print(f"? Address {addr}: {e}")
            time.sleep(0.1)  # Small delay between reads
                
    finally:
        client.close()

def main():
    print("ClickPlus PLC Modbus Connection Test")
    print("="*40)
    
    # Step 1: Test network connectivity
    if not test_network_connectivity():
        print("\nNetwork connectivity failed. Check:")
        print("1. Ethernet cable connection")
        print("2. IP address configuration on both devices")
        print("3. PLC network settings")
        return
    
    # Step 2: Test Modbus connection
    client = ModbusTcpClient(host=PLC_IP, port=PLC_PORT, timeout=TIMEOUT)
    
    try:
        print(f"\nConnecting to ClickPlus PLC at {PLC_IP}:{PLC_PORT}...")
        connection = client.connect()
        
        if not connection:
            print("Failed to connect to PLC")
            print("\nTroubleshooting steps:")
            print("1. Verify Modbus TCP is enabled on the PLC")
            print("2. Check PLC IP address and port settings")
            print("3. Ensure PLC is in 'Run' mode if required")
            print("4. Try different Unit ID (0, 1, 255)")
            test_multiple_addresses()
            return
        
        print("? Connected successfully!")
        
        # Test reading the specified address
        print(f"\nReading register at address {ADDRESS}...")
        result = client.read_holding_registers(address=ADDRESS, count=1, unit=UNIT_ID)
        
        if result.isError():
            print(f"? Modbus error: {result}")
            print("This address might not exist or be accessible")
            test_multiple_addresses()
        else:
            value = result.registers[0]
            print(f"? Register {ADDRESS} value: {value}")
            print(f"? Register {ADDRESS} value (hex): 0x{value:04X}")
            print(f"? Register {ADDRESS} value (binary): {bin(value)}")
        
    except Exception as e:
        print(f"? Error: {e}")
        print("\nThis might indicate:")
        print("1. Network connectivity issues")
        print("2. Incorrect PLC configuration")
        print("3. Firewall blocking the connection")
    
    finally:
        client.close()
        print("\nConnection closed")

if __name__ == "__main__":
    main()
