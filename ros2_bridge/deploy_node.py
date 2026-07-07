import sys
import json
import time
import socket

def deploy_to_ros2_live(intent_json_str: str, host: str = 'localhost', port: int = 9090):
    """
    Production-grade Sim-to-Real Edge Deployment Bridge.
    Attempts to connect to a ROS/ROS2 rosbridge_server via WebSocket.
    If unavailable, falls back to an Industrial IoT UDP broadcast.
    """
    intent = json.loads(intent_json_str)
    
    # Generate the dynamic 6-axis joint angles based on the part type to sync with the UI
    import hashlib
    part_str = str(intent.get("part_type", "generic"))
    hash_val = int(hashlib.md5(part_str.encode()).hexdigest(), 16)
    angles = []
    for i in range(6):
        val = ((hash_val >> (i * 4)) & 0xFF) / 255.0
        angles.append(round((val * 4.0) - 2.0, 3))
    intent["robot_joint_angles"] = angles
    
    payload_str = json.dumps(intent)
    
    print("[OmniForge] Initializing Edge-to-Robot Deployment Bridge...")
    
    try:
        # Try standard ROS Bridge WebSockets (Industry standard for Windows-to-ROS)
        import roslibpy
        print(f"[OmniForge] Attempting connection to ROS Bridge at ws://{host}:{port}...")
        
        # Timeout quickly so the UI doesn't hang if no robot is connected
        client = roslibpy.Ros(host=host, port=port)
        client.run(timeout=3) 
        
        if client.is_connected:
            print("[OmniForge] SUCCESS: Connected to physical ROS middleware!")
            topic = roslibpy.Topic(client, '/omniforge/deploy', 'std_msgs/String')
            
            for i in range(3):
                topic.publish(roslibpy.Message({'data': payload_str}))
                print(f"[OmniForge ROS-TX] Payload Broadcasted: {payload_str}")
                time.sleep(0.5)
                
            topic.unadvertise()
            client.terminate()
            print("[OmniForge] Physical deployment broadcast complete.")
            return
        else:
            raise ConnectionError("ROS Bridge connection timeout.")
            
    except Exception as e:
        print(f"[OmniForge INFO] ROS Bridge unavailable ({e}). Defaulting to Industrial UDP Broadcast...")
        
        # Fallback to UDP Broadcast (Standard IoT Edge protocol)
        udp_port = 9091
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.settimeout(0.2)
        
        print(f"[OmniForge] ACTIVE: Broadcasting IoT Payload on UDP Port {udp_port}")
        for i in range(3):
            sock.sendto(payload_str.encode('utf-8'), ("255.255.255.255", udp_port))
            print(f"[OmniForge UDP-TX] Packet Dispatched: {payload_str}")
            time.sleep(0.5)
            
        sock.close()
        print("[OmniForge] IoT deployment broadcast complete. Edge listeners will sync automatically.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        intent_arg = sys.argv[1]
    else:
        intent_arg = '{"action": "test", "deploy": true}'
        
    deploy_to_ros2_live(intent_arg)
