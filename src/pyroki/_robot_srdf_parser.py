from xml.etree import ElementTree as ET

def read_disabled_collisions_from_srdf(srdf_path):
    """Read disabled collision pairs from SRDF file."""
    
    tree = ET.parse(srdf_path)
    root = tree.getroot()
    
    disabled_pairs = []
    
    for disable_elem in root.findall('disable_collisions'):
        link1 = disable_elem.get('link1')
        link2 = disable_elem.get('link2')
        reason = disable_elem.get('reason', 'Unknown')
        
        disabled_pairs.append({
            'link1': link1,
            'link2': link2,
            'reason': reason
        })
    
    return disabled_pairs

def read_group_states_from_srdf(srdf_path):
    """Read named configurations from SRDF file."""
    
    tree = ET.parse(srdf_path)
    root = tree.getroot()
    
    group_states = {}
    
    for state_elem in root.findall('group_state'):
        group_name = state_elem.get('group')
        state_name = state_elem.get('name')
        
        joints = {}
        for joint_elem in state_elem.findall('joint'):
            joint_name = joint_elem.get('name')
            joint_value = float(joint_elem.get('value'))
            joints[joint_name] = joint_value
        
        if group_name not in group_states:
            group_states[group_name] = {}
        
        group_states[group_name][state_name] = joints
    
    return group_states

# Usage
if __name__ == "__main__":
    srdf_path = "resources/ur5/ur5.srdf"

    # Read disabled collisions
    disabled_pairs = read_disabled_collisions_from_srdf(srdf_path)
    print(f"Found {len(disabled_pairs)} disabled collision pairs:")
    for pair in disabled_pairs[:]:  # Show first 5
        print(f"  {pair['link1']} <-> {pair['link2']} (reason: {pair['reason']})")

    # Read group states
    group_states = read_group_states_from_srdf(srdf_path)
    print(f"\nFound group states: {list(group_states.keys())}")
    for group_name, states in group_states.items():
        print(f"\nGroup: {group_name}")
        for state_name, joints in states.items():
            print(f"  State '{state_name}': {joints}")